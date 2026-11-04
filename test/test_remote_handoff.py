# This module is part of GitPython and is released under the
# 3-Clause BSD License: https://opensource.org/license/bsd-3-clause/

"""Acceptance tests for nested, recoverable remote fetch/push handoff generations.

Every test here uses fully local repositories (temporary bare repositories and
clones over the file transport); no remote service is ever contacted. The
assertions watch cross-exit *time and causality on one and the same
* :class:`~git.repo.base.Repo` instance, not a ``FetchInfo``/``PushInfo`` flag
table:

* a fetch intent of generation ``N`` can no longer rewrite that repository's
  remote-tracking refs once the host advanced to ``N+1`` while the fetch was in
  flight; its footprints are bounced before
  :class:`~git.exc.StaleRemoteGenerationError` is raised;
* the stderr/``FetchInfo`` fetch path and the porcelain/``PushInfo`` push path
  fork onto independent ledgers inside one generation;
* despite returning ``FetchInfo`` objects, ``git pull`` lands its fetch half
  together with its merge drive on a third ledger -- its ``HEAD``/index/
  worktree/``ORIG_HEAD``/``MERGE_HEAD`` side effects are neither the bare fetch
  nor the push side effects;
* a short-lived nested generation that mounts a temporary remote on the same
  repository cannot bleed into an outer, still unfinished fetch or push
  intent, and bounces cleanly (mount, tracking namespace, pushed tip) when it
  fails.
"""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from git import (
    FetchInfo,
    PushInfo,
    Remote,
    RemoteGenerationScope,
    RemoteHandoff,
    RemoteHandoffLedger as Ledger,
    RemoteIntent,
    RemoteLanding,
    Repo,
    StaleRemoteGenerationError,
    remote_generation,
    remote_handoff,
)
from git.remote import RemoteProgress
from git.util import rmtree

L = Ledger


# { helpers


def _set_identity(repo):
    with repo.config_writer() as writer:
        writer.set_value("user", "name", "Handoff Tester")
        writer.set_value("user", "email", "handoff@example.com")


def _commit_file(repo, name, content):
    target = Path(repo.working_tree_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    repo.index.add([name])
    return repo.index.commit("Write " + name)


def _all_refs(repo):
    out = repo.git.for_each_ref(format="%(objectname) %(refname)")
    refs = {}
    for line in out.splitlines():
        sha, _, refname = line.partition(" ")
        if refname:
            refs[refname.strip()] = sha.strip()
    return refs


def _ref(repo, name):
    return _all_refs(repo).get(name)


def _ls_remote(repo, url):
    out = repo.git.ls_remote("--refs", url)
    refs = {}
    for line in out.splitlines():
        sha, _, refname = line.partition("\t")
        if refname:
            refs[refname.strip()] = sha.strip()
    return refs


def _read(path):
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


class _AdvanceOnProgress(RemoteProgress):
    """Open generation N+1 on the first progress BEGIN of an in-flight command."""

    def __init__(self, handoff):
        super().__init__()
        self._handoff = handoff
        self._cm = None
        self.scope = None

    def update(self, op_code, cur_count, max_count=None, message=""):
        if self._cm is None and op_code & RemoteProgress.BEGIN:
            self._cm = self._handoff.generation()
            self.scope = self._cm.__enter__()

    def close_new_generation(self):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None


def _ops(handoff_or_scope, ledger=None, op=None):
    landings = handoff_or_scope.landings
    if ledger is not None:
        landings = tuple(landing for landing in landings if landing.ledger is ledger)
    if op is not None:
        landings = tuple(landing for landing in landings if landing.op == op)
    return tuple((landing.generation, landing.ledger, landing.op, landing.name) for landing in landings)


# } END helpers


# { fixtures


@pytest.fixture
def cluster(tmp_path):
    """A bare upstream with two commits, a local clone and a second pusher clone."""
    upstream_path = tmp_path / "upstream.git"
    side_path = tmp_path / "side.git"
    seed_path = tmp_path / "seed"
    local_path = tmp_path / "local"
    other_path = tmp_path / "other"

    upstream = Repo.init(upstream_path, bare=True, initial_branch="master")
    upstream.git.symbolic_ref("HEAD", "refs/heads/master")
    side = Repo.init(side_path, bare=True, initial_branch="master")
    side.git.symbolic_ref("HEAD", "refs/heads/master")

    seed = Repo.init(seed_path, initial_branch="master")
    _set_identity(seed)
    _commit_file(seed, "a.txt", "one\n")
    _commit_file(seed, "b.txt", "two\n")
    seed.git.push(str(upstream.git_dir), "master:master")

    local = Repo.clone_from(str(upstream.git_dir), local_path, branch="master")
    other = Repo.clone_from(str(upstream.git_dir), other_path, branch="master")
    for repo in (local, other):
        _set_identity(repo)

    yield SimpleNamespace(
        tmp=tmp_path,
        upstream=upstream,
        side=side,
        upstream_url=str(upstream.git_dir),
        side_url=str(side.git_dir),
        seed=seed,
        local=local,
        other=other,
    )

    for repo in (local, other, seed, side, upstream):
        repo.close()
    with contextlib.suppress(OSError):
        rmtree(tmp_path)


def _advance_upstream(cluster, name, content):
    """Commit on the pusher clone and publish to the bare upstream."""
    commit = _commit_file(cluster.other, name, content)
    cluster.other.git.push("origin", "master:master")
    return commit.hexsha


# } END fixtures


# { generation lifecycle


def test_registry_is_shared_by_every_remote_handle_of_one_repo(cluster):
    repo = cluster.local
    handoff = repo.remote_handoff

    assert remote_handoff(repo) is handoff
    assert repo.remotes.origin.handoff is handoff
    assert repo.remote("origin").handoff is handoff
    assert Remote(repo, "origin").handoff is handoff
    assert isinstance(handoff, RemoteHandoff)
    assert not handoff.active
    assert handoff.current == 0
    assert handoff.frontier == 0
    assert handoff.landings == ()

    with repo.remote_generation() as scope:
        mounted = repo.create_remote("mounted", cluster.side_url)
        assert scope.number == 1
        assert handoff.current == 1
        assert handoff.active
        assert mounted.handoff is handoff
        assert Remote(repo, "mounted").handoff is handoff
    assert not handoff.active
    assert scope.closed
    assert scope.parent is None


def test_generation_is_declarable_progressive_and_nested(cluster):
    handoff = cluster.local.remote_handoff

    with handoff.generation() as first:
        assert first.number == 1
        assert first.frontier == 1
        with handoff.generation() as second:
            assert second.number == 2
            assert second.parent is first
        assert second.closed
        assert handoff.current == 1
        with handoff.generation(declare=5) as jumped:
            assert jumped.number == 5
            assert jumped.frontier == 5
        assert handoff.frontier == 5

    with pytest.raises(ValueError):
        with handoff.generation(declare=5):
            pass
    with pytest.raises(ValueError):
        with handoff.generation(declare=4):
            pass
    with pytest.raises(ValueError):
        with handoff.generation(declare=1.5):
            pass

    with handoff.generation(declare=6) as sixth:
        assert sixth.number == 6


def test_module_and_remote_context_managers_open_the_same_lineage(cluster):
    repo = cluster.local
    handoff = repo.remote_handoff

    with remote_generation(repo) as scope_a:
        assert isinstance(scope_a, RemoteGenerationScope)
        assert scope_a.number == 1
        with Remote.handoff_generation(repo) as scope_b:
            assert scope_b.number == 2
            assert scope_b.parent is scope_a
    assert handoff.frontier == 2


def test_registry_is_never_shared_between_repo_instances(cluster):
    local = cluster.local
    handoff = local.remote_handoff

    with handoff.generation():
        pass
    assert handoff.frontier == 1

    # The other clone is a different Repo instance and therefore has its own
    # lineage even though it reaches the same bare repository over the file
    # transport.
    other_handoff = cluster.other.remote_handoff
    assert other_handoff is not handoff
    assert remote_handoff(cluster.other) is other_handoff
    assert other_handoff.frontier == 0
    with other_handoff.generation() as scope:
        assert scope.number == 1
        assert handoff.current == 0
    assert handoff.frontier == 1


def test_legacy_intent_without_generation_is_a_noop(cluster):
    handoff = cluster.local.remote_handoff
    intent = handoff.claim(Ledger.FETCH, "origin")

    assert isinstance(intent, RemoteIntent)
    assert not intent.enabled
    assert intent.generation == 0
    assert intent.ledger is Ledger.FETCH
    assert intent.remote_name == "origin"
    assert intent.check("fetch", "origin") is False
    intent.remember_fetch("origin")
    intent.remember_push("origin")
    assert intent.land_fetch_infos(()) is None
    assert intent.land_push_infos([]) is None
    assert handoff.landings == ()


def test_intent_claimed_before_an_inner_advance_is_stale(cluster):
    handoff = cluster.local.remote_handoff

    with handoff.generation():
        outer_intent = handoff.claim(Ledger.MOUNT, "origin")
        with handoff.generation():
            with pytest.raises(StaleRemoteGenerationError):
                outer_intent.check("remote-add", "origin")


# } END generation lifecycle


# { required acceptance: stale fetch cannot land on the newer ledger


def test_stale_inflight_fetch_cannot_rewrite_remote_tracking(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    new_sha = _advance_upstream(cluster, "c.txt", "three\n")

    tracking_before = _ref(repo, "refs/remotes/origin/master")
    fetch_head_before = _read(Path(repo.git_dir, "FETCH_HEAD"))
    assert tracking_before != new_sha

    advance = _AdvanceOnProgress(handoff)
    with handoff.generation() as stale_scope:
        with pytest.raises(StaleRemoteGenerationError):
            origin.fetch(progress=advance)
    advance.close_new_generation()

    # The in-flight fetch did move refs inside git, but the stale fence bounced
    # every footprint before raising.
    assert _ref(repo, "refs/remotes/origin/master") == tracking_before
    assert _read(Path(repo.git_dir, "FETCH_HEAD")) == fetch_head_before
    assert advance.scope is not None and advance.scope.number == stale_scope.number + 1

    # Only the pre-flight "fetch" claim landed for the stale generation; no
    # parsed FetchInfo result did.
    stale_scope_ops = _ops(stale_scope)
    assert stale_scope_ops == ((stale_scope.number, Ledger.FETCH, "fetch", "origin"),)
    assert not _ops(stale_scope, Ledger.FETCH, "fetch-ref")

    # The newer ledger is perfectly usable: a fetch under the newer generation
    # moves the tracking ref and records its FetchInfo results there.
    with handoff.generation() as fresh_scope:
        origin.fetch()
    assert _ref(repo, "refs/remotes/origin/master") == new_sha
    assert any(isinstance(landing.detail, FetchInfo) for landing in fresh_scope.landings)
    assert all(landing.generation == fresh_scope.number for landing in fresh_scope.landings)


def test_failed_generation_restores_fetch_tracking_and_fetch_head(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    _advance_upstream(cluster, "c.txt", "three\n")
    _advance_upstream(cluster, "d.txt", "four\n")

    tracking_before = _ref(repo, "refs/remotes/origin/master")
    fetch_head_before = _read(Path(repo.git_dir, "FETCH_HEAD"))

    with pytest.raises(RuntimeError, match="boom"):
        with handoff.generation() as scope:
            origin.fetch()
            assert _ref(repo, "refs/remotes/origin/master") != tracking_before
            raise RuntimeError("boom")

    assert _ref(repo, "refs/remotes/origin/master") == tracking_before
    assert _read(Path(repo.git_dir, "FETCH_HEAD")) == fetch_head_before
    assert scope.closed
    # The audit trail keeps what the generation did before it rolled back.
    assert _ops(scope, Ledger.FETCH, "fetch-ref")


def test_settled_fetch_keeps_its_remote_tracking_update(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    new_sha = _advance_upstream(cluster, "c.txt", "three\n")

    with handoff.generation():
        origin.fetch()
    assert _ref(repo, "refs/remotes/origin/master") == new_sha


# }


# { required acceptance: fetch / push fork on independent ledgers


def test_fetch_and_push_fork_inside_one_generation(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    new_sha = _advance_upstream(cluster, "c.txt", "three\n")

    # A local-only branch to push to the bare upstream.
    repo.git.checkout("-b", "feature")
    feature_sha = _commit_file(repo, "f.txt", "feature\n").hexsha
    repo.git.checkout("master")

    with handoff.generation() as scope:
        fetch_infos = origin.fetch()
        push_infos = origin.push("feature:feature")

    assert _ref(repo, "refs/remotes/origin/master") == new_sha
    assert _ls_remote(repo, cluster.upstream_url)["refs/heads/feature"] == feature_sha
    assert all(isinstance(info, FetchInfo) for info in fetch_infos)
    assert all(isinstance(info, PushInfo) for info in push_infos)

    fetch_landings = [landing for landing in scope.landings if landing.ledger is Ledger.FETCH]
    pull_merge_landings = [landing for landing in scope.landings if landing.ledger is Ledger.PULL_MERGE]
    assert fetch_landings and _ops(scope, Ledger.FETCH, "fetch-ref")
    assert not pull_merge_landings
    assert all(isinstance(landing.detail, FetchInfo) for landing in scope.landings if landing.op == "fetch-ref")
    assert all(isinstance(landing.detail, PushInfo) for landing in scope.landings if landing.op == "push-ref")
    assert not any(
        landing.ledger is Ledger.FETCH and isinstance(landing.detail, PushInfo) for landing in scope.landings
    )
    assert not any(
        landing.ledger is Ledger.PUSH and isinstance(landing.detail, FetchInfo) for landing in scope.landings
    )
    assert _ops(scope, Ledger.PUSH, "push-ref") == ((scope.number, Ledger.PUSH, "push-ref", "refs/heads/feature"),)


def test_rejected_push_keeps_soft_error_chain_inside_the_generation(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    _advance_upstream(cluster, "c.txt", "three\n")  # local master is now behind.

    with handoff.generation() as scope:
        partial = origin.push("master:master")
        assert partial.error is not None
        assert _ops(scope, Ledger.PUSH, "push-error") == ((scope.number, Ledger.PUSH, "push-error", "origin"),)
        # The softer push error chain is attached, not raised: the generation
        # stays live and the host may decide to continue or abort.
        assert origin.fetch()

    assert scope.closed


def test_raising_push_error_outside_generation_body_bounces_the_generation(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    _advance_upstream(cluster, "c.txt", "three\n")

    repo.git.checkout("-b", "feature")
    _commit_file(repo, "f.txt", "feature\n")
    repo.git.checkout("master")
    tracking_before = _ref(repo, "refs/remotes/origin/master")

    with pytest.raises(Exception):  # noqa: B017 - raise_if_error attaches GitCommandError
        with handoff.generation():
            origin.fetch()
            origin.push("feature:feature")
            origin.push("master:master").raise_if_error()

    # The whole generation bounces: the earlier successful fetch and the
    # successful branch push are both undone, LIFO, together with the reject.
    assert _ref(repo, "refs/remotes/origin/master") == tracking_before
    assert "refs/heads/feature" not in _ls_remote(repo, cluster.upstream_url)


def test_stale_inflight_push_pushes_the_remote_tip_back(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff

    repo.git.checkout("-b", "feature")
    first = _commit_file(repo, "f.txt", "one\n").hexsha
    repo.git.checkout("master")
    with handoff.generation():
        origin.push("feature:feature")
    assert _ls_remote(repo, cluster.upstream_url)["refs/heads/feature"] == first

    repo.git.checkout("feature")
    second = _commit_file(repo, "g.txt", "two\n").hexsha
    repo.git.checkout("master")

    advance = _AdvanceOnProgress(handoff)
    with handoff.generation():
        with pytest.raises(StaleRemoteGenerationError):
            origin.push("feature:feature", progress=advance)
    advance.close_new_generation()

    remote = _ls_remote(repo, cluster.upstream_url)
    assert remote["refs/heads/feature"] == first
    assert remote["refs/heads/feature"] != second


# }


# { required acceptance: pull merge side effects fork from bare fetch


def test_pull_lands_fetch_infos_on_the_pull_merge_ledger(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    fetched_sha = _advance_upstream(cluster, "c.txt", "three\n")
    head_before = repo.head.commit.hexsha

    with handoff.generation() as scope:
        # Bare fetch first: tracking moves, HEAD does not.
        origin.fetch()
        assert repo.head.commit.hexsha == head_before
        assert _ref(repo, "refs/remotes/origin/master") == fetched_sha

        # The newer commit only reaches the bare upstream after the bare
        # fetch, so it is the pull -- not a fetch -- that brings it in.
        pulled_sha = _advance_upstream(cluster, "d.txt", "four\n")

        # Pull now: same FetchInfo object type, but it must not land on FETCH.
        pull_infos = origin.pull()
        assert all(isinstance(info, FetchInfo) for info in pull_infos)
        assert _ops(scope, Ledger.FETCH, "pull-ref") == ()
        pull_ref_ops = _ops(scope, Ledger.PULL_MERGE, "pull-ref")
        merge_ops = _ops(scope, Ledger.PULL_MERGE, "merge")
        assert pull_ref_ops and merge_ops

    # ``git pull`` (the pin command path) fast-forwards HEAD through
    # FETCH_HEAD and advances the tracking ref to the pulled tip as well;
    # what forks from the bare fetch is the ledger/merge drive, not the
    # objects -- and the same FetchInfo objects land on PULL_MERGE.
    assert repo.head.commit.hexsha == pulled_sha
    assert repo.git.rev_parse("FETCH_HEAD") == pulled_sha
    assert _ref(repo, "refs/remotes/origin/master") == pulled_sha
    assert Path(repo.git_dir, "ORIG_HEAD").exists()
    assert not Path(repo.git_dir, "MERGE_HEAD").exists()
    assert all(landing.generation == scope.number for landing in scope.landings)
    assert not _ops(scope, Ledger.PUSH)


def test_stale_inflight_pull_bounces_the_merge_drive(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    _advance_upstream(cluster, "c.txt", "three\n")

    head_before = repo.head.commit.hexsha
    tracking_before = _ref(repo, "refs/remotes/origin/master")
    orig_path = Path(repo.git_dir, "ORIG_HEAD")
    orig_before = _read(orig_path)

    advance = _AdvanceOnProgress(handoff)
    with handoff.generation():
        with pytest.raises(StaleRemoteGenerationError):
            origin.pull(progress=advance)
    advance.close_new_generation()

    assert repo.head.commit.hexsha == head_before
    assert _ref(repo, "refs/remotes/origin/master") == tracking_before
    assert not Path(repo.git_dir, "MERGE_HEAD").exists()
    assert _read(orig_path) == orig_before


def test_conflicting_pull_inside_failing_generation_is_aborted(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff

    # Same file changed differently on both sides since the common ancestor.
    _advance_upstream(cluster, "div.txt", "other side\n")
    local_commit = _commit_file(repo, "div.txt", "local side\n")
    tracking_before = _ref(repo, "refs/remotes/origin/master")
    orig_path = Path(repo.git_dir, "ORIG_HEAD")
    orig_before = _read(orig_path)

    with pytest.raises(Exception):  # noqa: B017 - git pull exits non-zero on conflict
        with handoff.generation():
            origin.pull()

    # The merge drive and its fetch half bounce together: the unfinished merge
    # is aborted (HEAD/index/worktree), the tracking ref returns, no residue.
    assert repo.head.commit.hexsha == local_commit.hexsha
    assert Path(repo.working_tree_dir, "div.txt").read_text(encoding="utf-8") == "local side\n"
    assert not Path(repo.git_dir, "MERGE_HEAD").exists()
    assert _ref(repo, "refs/remotes/origin/master") == tracking_before
    assert _read(orig_path) == orig_before


# }


# { required acceptance: nested temporary remote generations on one Repo


def test_nested_temp_remote_generation_settles_without_bleeding(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    new_sha = _advance_upstream(cluster, "c.txt", "three\n")

    # The side bare gets a master the temporary remote can fetch.
    cluster.other.git.push(cluster.side_url, "master:master")

    with handoff.generation() as outer:
        # Outer, unfinished fetch and push intents.
        origin.fetch()
        repo.git.checkout("-b", "outerfeat")
        _commit_file(repo, "o.txt", "outer\n")
        repo.git.checkout("master")
        origin.push("outerfeat:outerfeat")
        outer_tracking = _ref(repo, "refs/remotes/origin/master")
        assert outer_tracking == new_sha

        with handoff.generation() as inner:
            assert inner.parent is outer
            temp = repo.create_remote("tmp", cluster.side_url)
            assert temp.handoff is handoff  # temporary mount, same registry
            assert Remote(repo, "tmp").handoff is handoff
            temp.fetch()
            repo.git.checkout("outerfeat")
            _commit_file(repo, "i.txt", "inner\n")
            repo.git.checkout("master")
            temp.push("outerfeat:innerfeat")

        # The inner generation settled: its own effects attributed to its
        # number, but the outer origin ledger never saw them.
        assert inner.closed
        assert all(landing.generation == inner.number for landing in inner.landings)
        assert _ops(inner, Ledger.FETCH, "fetch-ref")
        assert _ops(inner, Ledger.PUSH, "push-ref")
        assert not [landing for landing in inner.landings if landing.name == "origin" and landing.op == "fetch-ref"]
        assert _ref(repo, "refs/remotes/origin/master") == outer_tracking
        assert _ref(repo, "refs/remotes/tmp/master") is not None
        side_refs = _ls_remote(repo, cluster.side_url)
        assert "refs/heads/innerfeat" in side_refs
        assert "tmp" in [remote.name for remote in repo.remotes]

        # Back in the outer generation, new work lands on the outer number.
        more_sha = _advance_upstream(cluster, "e.txt", "five\n")
        origin.fetch()
        assert _ref(repo, "refs/remotes/origin/master") == more_sha
        assert _ops(outer, Ledger.FETCH, "fetch-ref")[-1][0] == outer.number

    # Both generations settled: nothing is bounced away.
    assert "refs/heads/outerfeat" in _ls_remote(repo, cluster.upstream_url)
    assert "tmp" in [remote.name for remote in repo.remotes]


def test_failing_nested_generation_bounces_its_own_mount_tracking_and_tip(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    new_sha = _advance_upstream(cluster, "c.txt", "three\n")

    with handoff.generation() as outer:
        origin.fetch()
        outer_tracking = _ref(repo, "refs/remotes/origin/master")
        assert outer_tracking == new_sha

        with pytest.raises(RuntimeError, match="inner boom"):
            with handoff.generation() as inner:
                temp = repo.create_remote("tmp2", cluster.side_url)
                # Empty bare: push master first so the namespace exists.
                temp.push("master:master")
                repo.git.checkout("-b", "boom")
                _commit_file(repo, "z.txt", "boom\n")
                repo.git.checkout("master")
                temp.push("boom:boom")
                assert "refs/heads/boom" in _ls_remote(repo, cluster.side_url)
                raise RuntimeError("inner boom")

        # The inner failure unmounted the temporary remote, removed its
        # tracking namespace and pushed the side tip back; the outer origin
        # tracking it never touched is exactly where it was.
        assert inner.closed
        assert "tmp2" not in [remote.name for remote in repo.remotes]
        assert _ref(repo, "refs/remotes/tmp2/master") is None
        side_refs = _ls_remote(repo, cluster.side_url)
        assert "refs/heads/boom" not in side_refs
        assert _ref(repo, "refs/remotes/origin/master") == outer_tracking
        assert not [
            landing for landing in handoff.landings if landing.generation == inner.number and landing.name == "origin"
        ]

        # The outer generation is still live and usable on its own number.
        more_sha = _advance_upstream(cluster, "d.txt", "four\n")
        origin.fetch()
        assert _ref(repo, "refs/remotes/origin/master") == more_sha
        assert _ops(outer, Ledger.FETCH, "fetch-ref")[-1][0] == outer.number


# }


# { mount ledger (remote add / rm / rename)


def test_failed_generation_unmounts_a_remote_it_mounted(cluster):
    repo = cluster.local
    handoff = repo.remote_handoff

    with pytest.raises(RuntimeError, match="mount boom"):
        with handoff.generation() as scope:
            repo.create_remote("m1", cluster.side_url)
            assert "m1" in [remote.name for remote in repo.remotes]
            assert _ops(scope, Ledger.MOUNT, "remote-add")
            raise RuntimeError("mount boom")

    assert "m1" not in [remote.name for remote in repo.remotes]
    assert _ref(repo, "refs/remotes/m1/master") is None


def test_failed_removal_replays_the_mount_with_its_url(cluster):
    repo = cluster.local
    handoff = repo.remote_handoff

    with handoff.generation():
        repo.create_remote("m2", cluster.side_url)
    stored_url = repo.git.config("--get", "remote.m2.url")
    assert "m2" in [remote.name for remote in repo.remotes]

    with pytest.raises(RuntimeError, match="rm boom"):
        with handoff.generation():
            repo.delete_remote("m2")
            assert "m2" not in [remote.name for remote in repo.remotes]
            raise RuntimeError("rm boom")

    assert "m2" in [remote.name for remote in repo.remotes]
    assert repo.remote("m2").url == stored_url
    # The replayed mount keeps a working fetch refspec.
    cluster.other.git.push(cluster.side_url, "master:master")
    assert repo.remote("m2").fetch()
    assert _ref(repo, "refs/remotes/m2/master") is not None


def test_failed_rename_renames_the_mount_back(cluster):
    repo = cluster.local
    handoff = repo.remote_handoff

    with handoff.generation():
        repo.create_remote("m3", cluster.side_url)

    with pytest.raises(RuntimeError, match="rename boom"):
        with handoff.generation() as scope:
            repo.remote("m3").rename("m3b")
            assert "m3b" in [remote.name for remote in repo.remotes]
            assert _ops(scope, Ledger.MOUNT, "remote-rename")
            raise RuntimeError("rename boom")

    names = [remote.name for remote in repo.remotes]
    assert "m3" in names
    assert "m3b" not in names


# }


# { pin parity: without a generation the carriers are unchanged


def test_unmanaged_fetch_push_pull_have_no_generation_footprint(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff

    # Unmanaged fast-forward fetch.
    fetched_sha = _advance_upstream(cluster, "c.txt", "three\n")
    fetch_infos = origin.fetch()
    assert _ref(repo, "refs/remotes/origin/master") == fetched_sha
    assert all(isinstance(info, FetchInfo) for info in fetch_infos)

    # Unmanaged new-branch push.
    repo.git.checkout("-b", "pinfeat")
    feature_sha = _commit_file(repo, "p.txt", "pin\n").hexsha
    repo.git.checkout("master")
    push_infos = origin.push("pinfeat:pinfeat")
    assert _ls_remote(repo, cluster.upstream_url)["refs/heads/pinfeat"] == feature_sha
    assert all(isinstance(info, PushInfo) for info in push_infos)

    # Unmanaged rejected push keeps its soft .error chain.
    _advance_upstream(cluster, "d.txt", "four\n")
    rejected = origin.push("master:master")
    assert rejected.error is not None

    # Unmanaged fast-forward pull drives the merge through FETCH_HEAD.
    pull_sha = _advance_upstream(cluster, "e.txt", "five\n")
    pull_infos = origin.pull()
    assert repo.head.commit.hexsha == pull_sha
    assert repo.git.rev_parse("FETCH_HEAD") == pull_sha
    assert all(isinstance(info, FetchInfo) for info in pull_infos)

    assert handoff.landings == ()
    assert not handoff.active


def test_unmanaged_mount_operations_have_no_generation_footprint(cluster):
    repo = cluster.local
    handoff = repo.remote_handoff

    remote = repo.create_remote("pin", cluster.side_url)
    remote.rename("pinb")
    assert "pinb" in [item.name for item in repo.remotes]
    repo.delete_remote("pinb")
    assert "pinb" not in [item.name for item in repo.remotes]
    assert handoff.landings == ()


def test_landing_namedtuple_shape(cluster):
    repo = cluster.local
    origin = repo.remotes.origin
    handoff = repo.remote_handoff
    _advance_upstream(cluster, "c.txt", "three\n")

    with handoff.generation() as scope:
        origin.fetch()

    landing = handoff.landings[-1]
    assert isinstance(landing, RemoteLanding)
    assert landing.generation == scope.number
    assert landing.ledger is Ledger.FETCH
    assert landing.op == "fetch-ref"
    assert landing.name
    assert isinstance(landing.detail, FetchInfo)


# }
