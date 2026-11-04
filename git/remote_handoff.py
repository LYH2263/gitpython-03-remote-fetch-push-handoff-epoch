# Copyright (C) 2026 Michael Trier (mtrier@gmail.com) and contributors
#
# This module is part of GitPython and is released under the
# 3-Clause BSD License: https://opensource.org/license/bsd-3-clause/

"""Nested, recoverable handoff generations for remote reference fetch and push.

GitPython keeps the effects of the remote commands on ledgers that historically
did not share a generation boundary:

* the *fetch* ledger: the remote-tracking refs under ``refs/remotes/`` (and
  followed tags) rewritten by :manpage:`git-fetch(1)`, parsed from stderr into
  :class:`~git.remote.FetchInfo` (``NEW_TAG``, ``NEW_HEAD``, ``TAG_UPDATE``,
  ``FAST_FORWARD``, ``HEAD_UPTODATE`` ...);
* the *push* ledger: the tips on the *other* repository rewritten by
  :manpage:`git-push(1)`, parsed from porcelain output into
  :class:`~git.remote.PushInfo`/:class:`~git.remote.PushInfoList`
  (``DELETED``, ``REMOTE_REJECTED``, ``UP_TO_DATE``, ``FORCED_UPDATE`` ...),
  a different flag table and a softer error chain (failures may be attached to
  ``PushInfoList.error`` instead of being raised);
* the *pull-merge* ledger: the merge drive of :manpage:`git-pull(1)`. Although
  ``pull`` returns ``FetchInfo`` objects, it goes through the ``git pull``
  command path: it moves ``HEAD`` (fast-forward or merge commit), the index and
  the worktree, may leave ``MERGE_HEAD`` on conflicts and writes ``ORIG_HEAD``.
  It is neither the bare-fetch exit nor the push exit;
* the *mount* ledger: the ``remote.<name>`` configuration written by
  ``git remote add``/``rm``/``rename`` (i.e. :meth:`Repo.create_remote
  <git.repo.base.Repo.create_remote>` and friends), together with the
  remote-tracking namespace owned by that name.

This module adds *handoff generations* on top of those ledgers. A generation is
a short-lived scope owned by a host (an embedding application). Every mutation
started inside a generation carries an :class:`RemoteIntent` stamped with that
generation. Once the host advances to a newer generation, intents of the older
generation can no longer land on this repository's newer ledger -- neither a
fetch's remote-tracking rewrite, nor a push's remote tip update, nor a pull's
merge drive; they fail with :class:`~git.exc.StaleRemoteGenerationError`.
For operations that were already in flight, the footprints the git command did
leave are bounced (tracking refs restored, pushed tips pushed back over a local
transport, merge reset/aborted) *before* the exception is raised.

Generations nest. A nested short-lived generation -- for example an inner task
that :meth:`creates <git.repo.base.Repo.create_remote>` a temporary remote on the
same :class:`~git.repo.base.Repo` and fetches from it while an outer fetch or
push is still unfinished -- gets its own generation number, its own landing log
and its own undo log: it can neither move the outer cursor nor leave its
temporary remote, tracking refs or pushed tips behind when it fails. The
registry is bound to one :class:`~git.repo.base.Repo` instance, so every
:class:`~git.remote.Remote` obtained from that repository (``repo.remotes``,
``repo.remote()``, ``repo.create_remote()`` or a temporary ``Remote(repo, ...)``)
shares the same lineage, while repositories that are not the same instance
never do.

When no generation is active every checkpoint is a no-op, leaving the behaviour
of :meth:`Remote.fetch <git.remote.Remote.fetch>`, :meth:`Remote.pull
<git.remote.Remote.pull>`, :meth:`Remote.push <git.remote.Remote.push>` and the
remote mount points exactly as they were without this module. No constructor
gains a required parameter.

The four ledgers are fenced independently. The fetch and push result paths
(:class:`~git.remote.FetchInfo` versus :class:`~git.remote.PushInfo`) are never
folded into one switch, and the pull merge drive is folded into neither: the
``FetchInfo`` list returned by ``pull`` is stamped on the ``PULL_MERGE``
ledger, not on ``FETCH``.
"""

__all__ = [
    "RemoteHandoffLedger",
    "RemoteLanding",
    "RemoteGenerationScope",
    "RemoteIntent",
    "RemoteHandoff",
    "remote_handoff",
    "remote_generation",
]

import contextlib
import enum
import logging
import os
import os.path as osp
import weakref
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
)

from git.exc import GitCommandError, StaleRemoteGenerationError

if TYPE_CHECKING:
    from git.remote import FetchInfo
    from git.repo import Repo

_logger = logging.getLogger(__name__)


class RemoteHandoffLedger(enum.Enum):
    """The independent ledgers a remote mutation can land on.

    :ivar FETCH:
        The remote-tracking ref rewrite (and followed tags) of a bare
        :manpage:`git-fetch(1)`, parsed into :class:`~git.remote.FetchInfo`.

    :ivar PUSH:
        The remote-side tip update performed by :manpage:`git-push(1)`, parsed
        into :class:`~git.remote.PushInfo` on the porcelain path. This is a
        different flag table, parser and error chain than :attr:`FETCH`.

    :ivar PULL_MERGE:
        The merge drive of :manpage:`git-pull(1)` -- ``HEAD``/index/worktree
        movement, ``ORIG_HEAD`` and possible ``MERGE_HEAD`` -- plus the
        fetch half that belongs to the same ``git pull`` invocation. The
        returned :class:`~git.remote.FetchInfo` objects still land here rather
        than on :attr:`FETCH`, because the command exit is not the bare-fetch
        exit.

    :ivar MOUNT:
        The ``remote.<name>`` configuration and tracking-namespace ownership
        changed by mounting (``git remote add``), unmounting (``git remote
        rm``) or renaming a remote.
    """

    FETCH = "fetch"
    PUSH = "push"
    PULL_MERGE = "pull-merge"
    MOUNT = "mount"


class RemoteLanding(NamedTuple):
    """A record of one mutation that passed a generation fence."""

    generation: int
    ledger: RemoteHandoffLedger
    op: str
    name: Optional[str]
    detail: Any = None


class _Undo:
    """A rollback action that can be consumed exactly once."""

    __slots__ = ("fn", "done")

    def __init__(self, fn: Callable[[], None]) -> None:
        self.fn = fn
        self.done = False


class _PullState(NamedTuple):
    """Pre-pull state needed to bounce the merge drive and its fetch half."""

    old_head: str
    had_merge_head: bool
    orig_head: Optional[bytes]


# { Snapshot primitives


def _read_file(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fp:
            return fp.read()
    except OSError:
        return None


def _write_or_remove(path: str, data: Optional[bytes]) -> None:
    if data is None:
        with contextlib.suppress(OSError):
            os.remove(path)
        return
    directory = osp.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(data)


def _all_refs(repo: "Repo") -> Dict[str, str]:
    """Map every local ref name to its object name, in one plumbing call."""
    out = repo.git.for_each_ref(format="%(objectname) %(refname)")
    refs: Dict[str, str] = {}
    assert isinstance(out, str)
    for line in out.splitlines():
        sha, _, refname = line.partition(" ")
        if refname:
            refs[refname.strip()] = sha.strip()
    return refs


def _restore_refs(repo: "Repo", previous: Dict[str, str], remote_name: Optional[str]) -> None:
    """Best-effort restore of every ref that moved away from `previous`."""
    tracking_prefix = "refs/remotes/%s/" % remote_name if remote_name else None
    with contextlib.suppress(Exception):
        current = _all_refs(repo)
    for refname, sha in previous.items():
        if tracking_prefix is not None and refname.startswith(tracking_prefix):
            # The mount itself may have been removed in the same rollback;
            # leave the namespace to `git remote rm` in that case.
            if not repo.config_reader().has_section('remote "%s"' % remote_name):
                continue
        if current.get(refname) != sha:
            with contextlib.suppress(Exception):
                repo.git.update_ref("--no-deref", refname, sha)
    for refname in current.keys() - previous.keys():
        if tracking_prefix is not None and refname.startswith(tracking_prefix):
            if not repo.config_reader().has_section('remote "%s"' % remote_name):
                continue
        with contextlib.suppress(Exception):
            repo.git.update_ref("-d", "--no-deref", refname)


def _ls_remote(repo: "Repo", url_or_name: str) -> Dict[str, str]:
    out = repo.git.ls_remote("--refs", url_or_name)
    refs: Dict[str, str] = {}
    assert isinstance(out, str)
    for line in out.splitlines():
        sha, _, refname = line.partition("\t")
        if refname:
            refs[refname.strip()] = sha.strip()
    return refs


def _push_url(repo: "Repo", remote_name: str) -> str:
    key = "remote.%s" % remote_name
    try:
        value = repo.git.config("--get", "%s.pushurl" % key)
        if value:
            assert isinstance(value, str)
            return value.strip()
    except GitCommandError:
        pass
    value = repo.git.config("--get", "%s.url" % key)
    assert isinstance(value, str)
    return value.strip()


def _config_get_all(repo: "Repo", key: str) -> List[str]:
    try:
        out = repo.git.config("--get-all", key)
    except GitCommandError:
        return []
    if not out:
        return []
    assert isinstance(out, str)
    return [line for line in out.splitlines() if line]


def _push_refspecs_back(repo: "Repo", url: str, specs: Sequence[str]) -> None:
    if not specs:
        return
    try:
        repo.git.push("--force", "--", url, *specs)
        return
    except Exception as ex:  # noqa: BLE001 - recovery must keep trying per ref
        _logger.warning("Batched generation rollback push failed; retrying each ref: %s", ex)
    for spec in specs:
        try:
            repo.git.push("--force", "--", url, spec)
        except Exception as ex:  # noqa: BLE001 - rollback must keep going
            # The old object may not be available locally (e.g. a remote ref
            # deleted before this repository ever fetched it). Nothing else
            # can be done without contacting the remote out of band.
            _logger.error("Could not restore remote ref %r during generation rollback: %s", spec, ex)


# } END snapshot primitives


class RemoteGenerationScope:
    """One open generation on a :class:`RemoteHandoff`.

    Instances are produced by :meth:`RemoteHandoff.generation`; they are not
    meant to be constructed directly.
    """

    __slots__ = ("_state", "_number", "_parent", "_frontier", "_undos", "_landings", "_closed")

    def __init__(
        self,
        state: "RemoteHandoff",
        number: int,
        frontier: int,
        parent: "Optional[RemoteGenerationScope]",
    ) -> None:
        self._state = state
        self._number = number
        self._parent = parent
        #: Frontier this generation's intents stamp; a later advance makes
        #: intents carrying an older frontier stale.
        self._frontier = frontier
        self._undos: List[_Undo] = []
        self._landings: List[RemoteLanding] = []
        self._closed = False

    #: The generation number of this scope.
    @property
    def number(self) -> int:
        return self._number

    #: The scope that was active when this one was opened, if any.
    @property
    def parent(self) -> "Optional[RemoteGenerationScope]":
        return self._parent

    #: The frontier captured when this scope opened.
    @property
    def frontier(self) -> int:
        return self._frontier

    #: ``True`` after the scope has settled or rolled back.
    @property
    def closed(self) -> bool:
        return self._closed

    #: All landings recorded within this scope, in order.
    @property
    def landings(self) -> Tuple[RemoteLanding, ...]:
        return tuple(self._landings)

    def _record(self, landing: RemoteLanding) -> None:
        self._landings.append(landing)
        self._state._record_landing(landing)

    def _defer(self, fn: Callable[[], None]) -> _Undo:
        undo = _Undo(fn)
        self._undos.append(undo)
        return undo

    def _settle(self) -> None:
        self._closed = True
        parent = self._parent
        if parent is not None and not parent._closed:
            # A cleanly finishing nested scope hands its still-open footprints
            # to the enclosing generation, so an outer rollback covers them;
            # landing attribution stays with this scope's number.
            parent._undos.extend(self._undos)

    def _rollback(self) -> None:
        """Undo every side effect still owned by this scope, last one first."""
        self._closed = True
        undos, self._undos = self._undos, []
        errors: List[BaseException] = []
        for undo in reversed(undos):
            if undo.done:
                continue
            undo.done = True
            try:
                undo.fn()
            except Exception as ex:  # noqa: BLE001 - rollback must keep going
                errors.append(ex)
                _logger.error("Remote handoff rollback action failed", exc_info=True)
        if errors:
            _logger.error(
                "Remote handoff generation %d rollback completed with %d failed action(s)",
                self._number,
                len(errors),
            )

    def __repr__(self) -> str:
        return "%s(number=%d, closed=%s)" % (type(self).__name__, self._number, self._closed)


class RemoteIntent:
    """A fetch/push/pull/mount intention stamped with a generation.

    An intent is created when a carrier starts (``Remote.fetch``,
    ``Remote.push``, ``Remote.pull`` or a remote add/remove/rename) and checked
    both before the git command is spawned and at the point where its parsed
    result settles. Intents created while no generation is active are *legacy*
    intents whose checkpoints are no-ops, which keeps the unmanaged behaviour
    identical to the plain code.
    """

    __slots__ = ("_state", "_scope", "_ledger", "_frontier", "_remote", "_undos")

    def __init__(
        self,
        state: "RemoteHandoff",
        scope: Optional[RemoteGenerationScope],
        ledger: RemoteHandoffLedger,
        frontier: int,
        remote_name: Optional[str] = None,
    ) -> None:
        self._state = state
        self._scope = scope
        self._ledger = ledger
        self._frontier = frontier
        self._remote = remote_name
        self._undos: List[_Undo] = []

    #: ``True`` for intents bound to an actual open generation.
    @property
    def enabled(self) -> bool:
        return self._scope is not None

    #: The generation number this intent belongs to, or ``0`` when legacy.
    @property
    def generation(self) -> int:
        return self._scope._number if self._scope is not None else 0

    #: The ledger this intent lands on.
    @property
    def ledger(self) -> RemoteHandoffLedger:
        return self._ledger

    #: Name of the remote this intent operates on, if known.
    @property
    def remote_name(self) -> Optional[str]:
        return self._remote

    def _staleness_reason(self) -> Optional[str]:
        scope = self._scope
        if scope is None:
            return None
        if scope._closed:
            return "closed"
        if self._state.current_scope() is not scope:
            return "superseded"
        if self._state.frontier != self._frontier:
            return "advanced"
        return None

    def _landing(self, op: str, name: Optional[str] = None, detail: Any = None) -> None:
        assert self._scope is not None
        landing = RemoteLanding(self._scope._number, self._ledger, op, name, detail)
        self._scope._record(landing)

    def check(self, op: str, name: Optional[str] = None, detail: Any = None) -> bool:
        """Fence a landing point that has not mutated state yet.

        :return:
            ``True`` if the landing belongs to a live generation and was
            recorded; ``False`` for legacy intents (no generation active), in
            which case the caller performs the plain, unmanaged action.

        :raise StaleRemoteGenerationError:
            If this intent's generation has settled, been rolled back, been
            covered by a nested newer generation, or been advanced past.
        """
        scope = self._scope
        if scope is None:
            return False
        reason = self._staleness_reason()
        if reason is not None:
            raise StaleRemoteGenerationError(
                "remote handoff generation %d cannot land %s on ledger %s: generation is %s"
                % (scope._number, op, self._ledger.value, reason)
            )
        self._landing(op, name, detail)
        return True

    def defer_undo(self, fn: Callable[[], None]) -> None:
        """Register a raw rollback action for this intent's generation.

        No-op for legacy intents. The action must run through raw plumbing
        (``repo.git.*``), never through a fenced carrier: rollback happens
        after the generation is already being torn down.
        """
        scope = self._scope
        if scope is None or scope._closed:
            return
        undo = scope._defer(fn)
        self._undos.append(undo)

    def _bounce(self) -> None:
        """Undo just this intent's own footprints, newest first.

        Used when an in-flight command discovers after the fact that its
        generation was superseded: earlier successful operations of the same
        generation stay intact.
        """
        undos, self._undos = self._undos, []
        for undo in reversed(undos):
            if undo.done:
                continue
            undo.done = True
            try:
                undo.fn()
            except Exception as ex:  # noqa: BLE001 - bounce must keep going
                _logger.error("Remote handoff stale-intent bounce failed", exc_info=True)
                _logger.error("... %s", ex)

    def _fence_after_flight(self, op: str) -> None:
        """Post-command fence: bounce own footprints, then raise if stale."""
        scope = self._scope
        if scope is None:
            return
        reason = self._staleness_reason()
        if reason is not None:
            self._bounce()
            raise StaleRemoteGenerationError(
                "remote handoff generation %d cannot settle %s on ledger %s: generation is %s"
                % (scope._number, op, self._ledger.value, reason)
            )

    # { Footprint snapshots

    def remember_fetch(self, remote_name: str) -> None:
        """Snapshot local refs and ``FETCH_HEAD`` before a managed fetch."""
        if self._scope is None or self._scope._closed:
            return
        repo = self._state.repo
        previous = _all_refs(repo)
        fetch_head_path = osp.join(os.fspath(repo.git_dir), "FETCH_HEAD")
        previous_fetch_head = _read_file(fetch_head_path)

        def undo() -> None:
            _restore_refs(repo, previous, remote_name)
            _write_or_remove(fetch_head_path, previous_fetch_head)

        self.defer_undo(undo)

    def remember_push(self, remote_name: str) -> None:
        """Snapshot the remote tip map (over the push URL) before a push."""
        if self._scope is None or self._scope._closed:
            return
        repo = self._state.repo
        url = _push_url(repo, remote_name)
        try:
            previous = _ls_remote(repo, url)
        except Exception as ex:  # noqa: BLE001 - snapshot failure must not block
            _logger.warning("Could not snapshot remote %r for generation rollback: %s", remote_name, ex)
            return

        def undo() -> None:
            try:
                current = _ls_remote(repo, url)
            except Exception as ex:  # noqa: BLE001
                _logger.error("Could not re-inspect remote %r during rollback: %s", remote_name, ex)
                return
            specs: List[str] = []
            for refname, new_sha in current.items():
                old_sha = previous.get(refname)
                if old_sha is None:
                    specs.append(":%s" % refname)  # The remote ref is new: delete it.
                elif old_sha != new_sha:
                    specs.append("%s:%s" % (old_sha, refname))  # Move the tip back.
            for refname, old_sha in previous.items():
                if refname not in current:
                    specs.append("%s:%s" % (old_sha, refname))  # Remote ref was deleted.
            _push_refspecs_back(repo, url, specs)

        self.defer_undo(undo)

    def remember_pull(self, remote_name: str) -> Optional[_PullState]:
        """Snapshot the fetch footprint *and* the merge drive before a pull."""
        if self._scope is None or self._scope._closed:
            return None
        repo = self._state.repo
        # Register the ref/FETCH_HEAD undo first so it runs after the head undo.
        self.remember_fetch(remote_name)
        old_head = repo.head.commit.hexsha
        merge_head_path = osp.join(os.fspath(repo.git_dir), "MERGE_HEAD")
        orig_head_path = osp.join(os.fspath(repo.git_dir), "ORIG_HEAD")
        state = _PullState(
            old_head=old_head,
            had_merge_head=osp.exists(merge_head_path),
            orig_head=_read_file(orig_head_path),
        )

        def undo() -> None:
            if osp.exists(merge_head_path) and not state.had_merge_head:
                # git pull stopped on conflicts: abort the unfinished merge,
                # which restores HEAD, index and worktree together.
                try:
                    repo.git.merge("--abort")
                except Exception as ex:  # noqa: BLE001
                    _logger.warning("git merge --abort failed; falling back to reset: %s", ex)
                    repo.git.reset("--hard", state.old_head)
            else:
                repo.git.reset("--hard", state.old_head)
            _write_or_remove(orig_head_path, state.orig_head)

        self.defer_undo(undo)
        return state

    def remember_remote_add(self, remote_name: str) -> None:
        """Arrange unmounting a remote that a managed generation mounted."""
        if self._scope is None or self._scope._closed:
            return
        repo = self._state.repo

        def undo() -> None:
            with contextlib.suppress(GitCommandError):
                repo.git.remote("rm", remote_name)

        self.defer_undo(undo)

    def remember_remote_remove(self, remote_name: str) -> None:
        """Snapshot a remote's configuration so a removal can be replayed back."""
        if self._scope is None or self._scope._closed:
            return
        repo = self._state.repo
        urls = _config_get_all(repo, "remote.%s.url" % remote_name)
        fetch_specs = _config_get_all(repo, "remote.%s.fetch" % remote_name)
        if not urls:
            return

        def undo() -> None:
            try:
                repo.git.remote("add", "--", remote_name, urls[0])
                for extra_url in urls[1:]:
                    repo.git.remote("set-url", "--add", "--", remote_name, extra_url)
                with contextlib.suppress(GitCommandError):
                    repo.git.config("--unset-all", "remote.%s.fetch" % remote_name)
                for spec in fetch_specs:
                    repo.git.config("--add", "remote.%s.fetch" % remote_name, spec)
            except Exception as ex:  # noqa: BLE001 - rollback must keep going
                _logger.error("Could not restore remote %r during rollback: %s", remote_name, ex)

        self.defer_undo(undo)

    def remember_remote_rename(self, old_name: str, new_name: str) -> None:
        """Arrange renaming a remotely back if a generation rolls back."""
        if self._scope is None or self._scope._closed:
            return
        repo = self._state.repo

        def undo() -> None:
            with contextlib.suppress(GitCommandError):
                repo.git.remote("rename", new_name, old_name)

        self.defer_undo(undo)

    # } END footprint snapshots

    # { Result landing

    def land_fetch_infos(self, infos: Sequence["FetchInfo"], op: str = "fetch") -> None:
        """Fence and record the parsed :class:`~git.remote.FetchInfo` result.

        Stamped on this intent's ledger (``FETCH`` for a bare fetch,
        ``PULL_MERGE`` for a pull): the same object type forks by ledger here.
        If the generation advanced while the command was in flight, the
        tracked-ref footprints are bounced before raising.
        """
        if self._scope is None:
            return
        self._fence_after_flight("%s-ref" % op)
        for info in infos:
            self._landing("%s-ref" % op, getattr(info, "name", None), info)

    def land_pull_head(self, state: Optional[_PullState]) -> None:
        """Record the merge drive of a completed pull on the PULL_MERGE ledger."""
        if self._scope is None or state is None:
            return
        self._fence_after_flight("merge")
        repo = self._state.repo
        new_head = repo.head.commit.hexsha
        if osp.exists(osp.join(os.fspath(repo.git_dir), "MERGE_HEAD")):
            self._landing("merge-conflict", "HEAD", (state.old_head, new_head))
        elif new_head != state.old_head:
            self._landing("merge", "HEAD", (state.old_head, new_head))

    def land_push_infos(self, push_infos: Any) -> None:
        """Fence and record the parsed :class:`~git.remote.PushInfo` result.

        A partial push failure already attached by the porcelain parser to
        ``PushInfoList.error`` is recorded as its own push-ledger landing; the
        softer push error chain is therefore not the same exit as a fetch
        failure, but it is still inside the generation.
        """
        if self._scope is None:
            return
        self._fence_after_flight("push-ref")
        for info in push_infos:
            self._landing("push-ref", info.remote_ref_string, info)
        error = getattr(push_infos, "error", None)
        if error is not None:
            self._landing("push-error", self._remote, error)

    # } END result landing


class RemoteHandoff:
    """Generation registry bound to one :class:`~git.repo.base.Repo` instance.

    Obtain an instance through :func:`remote_handoff` or
    :attr:`Repo.remote_handoff <git.repo.base.Repo.remote_handoff>`; the
    registry is shared by every :class:`~git.remote.Remote` of that repository
    instance (``repo.remotes``, ``repo.remote()``, ``repo.create_remote()`` and
    temporary ``Remote(repo, ...)`` handles). It is never shared with a
    different :class:`~git.repo.base.Repo` instance.
    """

    __slots__ = ("_repo", "_frontier", "_top", "_landings")

    def __init__(self, repo: "Repo") -> None:
        self._repo = weakref.ref(repo)
        self._frontier = 0
        self._top: Optional[RemoteGenerationScope] = None
        self._landings: List[RemoteLanding] = []

    #: The repository this registry belongs to.
    @property
    def repo(self) -> "Repo":
        repo = self._repo()
        assert repo is not None, "RemoteHandoff outlived its repository"
        return repo

    #: Highest generation number ever opened on this registry.
    @property
    def frontier(self) -> int:
        return self._frontier

    #: Number of the currently open (innermost) generation, or ``0``.
    @property
    def current(self) -> int:
        return self._top._number if self._top is not None else 0

    #: Whether any generation scope is currently open.
    @property
    def active(self) -> bool:
        return self._top is not None

    def current_scope(self) -> Optional[RemoteGenerationScope]:
        """The innermost open scope, or ``None`` when no generation is active."""
        top = self._top
        return top if top is not None and not top._closed else None

    #: Every landing recorded on this registry, in order, across generations.
    @property
    def landings(self) -> Tuple[RemoteLanding, ...]:
        return tuple(self._landings)

    def _record_landing(self, landing: RemoteLanding) -> None:
        self._landings.append(landing)

    @contextlib.contextmanager
    def generation(
        self,
        declare: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Iterator[RemoteGenerationScope]:
        """Open a new remote handoff generation.

        :param declare:
            Declare an explicit generation number. It must be strictly greater
            than the registry's current frontier; numbers may be skipped. When
            ``None`` (the default), the frontier is advanced by one.

        :param name:
            Optional descriptive label, reserved for debugging.

        On normal exit the generation settles: effects that landed inside it
        remain (a nested generation hands them to its enclosing generation).
        On exception the generation bounces every side effect recorded through
        its intents -- restored remote-tracking refs, pushed-back remote tips,
        aborted/reset merges and unmounted temporary remotes -- in reverse
        landing order, and the exception propagates.
        """
        if declare is None:
            number = self._frontier + 1
        else:
            if not isinstance(declare, int) or isinstance(declare, bool) or declare <= self._frontier:
                raise ValueError(
                    "declared remote handoff generation %r must be an integer greater than the current frontier %d"
                    % (declare, self._frontier)
                )
            number = declare
        # END handle explicit declaration
        self._frontier = number
        scope = RemoteGenerationScope(self, number, number, self._top)
        self._top = scope
        try:
            yield scope
        except BaseException:
            scope._rollback()
            raise
        else:
            scope._settle()
        finally:
            if self._top is scope:
                self._top = scope._parent

    def claim(self, ledger: RemoteHandoffLedger, remote_name: Optional[str] = None) -> RemoteIntent:
        """Stamp an intent for *ledger* at the current generation.

        Returns a legacy no-op intent when no generation is active. The intent
        remembers the generation (and frontier) it was claimed in, so it can
        later detect that the host advanced to a newer generation before the
        command settled.
        """
        top = self.current_scope()
        if top is None:
            return RemoteIntent(self, None, ledger, self._frontier, remote_name)
        return RemoteIntent(self, top, ledger, self._frontier, remote_name)


_registry: "weakref.WeakKeyDictionary[Repo, RemoteHandoff]" = weakref.WeakKeyDictionary()


def remote_handoff(repo: "Repo") -> RemoteHandoff:
    """Return the :class:`RemoteHandoff` registry for *repo*.

    One registry exists per repository instance; it is created lazily and
    garbage collected together with the repository.
    """
    handoff = _registry.get(repo)
    if handoff is None:
        handoff = RemoteHandoff(repo)
        _registry[repo] = handoff
    return handoff


@contextlib.contextmanager
def remote_generation(repo: "Repo", declare: Optional[int] = None) -> Iterator[RemoteGenerationScope]:
    """Convenience wrapper for ``remote_handoff(repo).generation(declare=declare)``."""
    with remote_handoff(repo).generation(declare=declare) as scope:
        yield scope
