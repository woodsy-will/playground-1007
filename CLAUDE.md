# Claude Code Instructions

## Git Workflow

- **Always `git fetch origin` and check branch state before starting any task.** Verify whether prior PRs have been merged and whether the working branch still exists or has diverged.
- Use **one branch per feature/issue**. Create from latest `main`. Never reuse a branch after its PR is merged.
- After a PR is merged by the user, start a new branch from the updated `main` for the next task.
- Never push new commits to a branch whose PR has already been merged.
