import os
import sys
import subprocess

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

updated = False

# Method 1: Try using git command-line tool (fastest and most reliable on Colab/Linux)
try:
    git_check = subprocess.run(["git", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if git_check.returncode == 0:
        fetch_res = subprocess.run(["git", "fetch", "origin"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        pull_res = subprocess.run(["git", "pull", "--ff-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if pull_res.returncode == 0:
            print(f"[Update] Git CLI: {pull_res.stdout.strip()}")
            updated = True
        else:
            err_msg = f"{pull_res.stderr.strip()} {pull_res.stdout.strip()}".strip()
            print(f"[Update] Git CLI update failed: {err_msg}")
            if os.path.isdir("/content"):
                branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE, text=True, check=False)
                curr_branch = branch_res.stdout.strip() or "anima-support"
                reset_res = subprocess.run(["git", "reset", "--hard", f"origin/{curr_branch}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                if reset_res.returncode == 0:
                    print(f"[Update] Git CLI auto-synced with origin/{curr_branch}")
                    updated = True
except Exception as e:
    pass

# Method 2: Fallback to pygit2 if git CLI didn't succeed and pygit2 is available
if not updated:
    try:
        import pygit2
        pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

        repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

        branch_name = repo.head.shorthand

        remote_name = 'origin'
        remote = repo.remotes[remote_name]

        remote.fetch()

        local_branch_ref = f'refs/heads/{branch_name}'
        local_branch = repo.lookup_reference(local_branch_ref)

        remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
        remote_commit = repo.revparse_single(remote_reference)

        merge_result, _ = repo.merge_analysis(remote_commit.id)

        if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
            print("[Update] Already up-to-date")
            updated = True
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
            local_branch.set_target(remote_commit.id)
            repo.head.set_target(remote_commit.id)
            repo.checkout_tree(repo.get(remote_commit.id))
            repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
            print("[Update] Fast-forward merge succeeded")
            updated = True
        elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
            print("[Update] Update failed - Did you modify any file?")
    except Exception as e:
        print(f"[Update] pygit2 check skipped or unavailable: {e}")

from launch import *

