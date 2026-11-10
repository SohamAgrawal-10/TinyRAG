# Git Basics

## What Is Git?

Git is a distributed version control system used to track changes in files. It allows developers to maintain a history of their work, compare versions, and collaborate on software projects.

## Commits

A commit records a snapshot of the staged changes in a Git repository. Each commit has a unique identifier and a message describing the changes.

Use `git add` to stage changes and `git commit` to create a commit.

## Branches

A branch provides an independent line of development. Developers can use branches to work on features or fixes without directly changing the main branch.

Use `git branch` to inspect existing branches.

Use `git switch -c <branch-name>` to create a new branch and switch to it.

Use `git switch <branch-name>` to switch to an existing branch.

## Remote Repositories

A remote repository is a version of a project hosted elsewhere, such as on GitHub. It allows developers to share their work and synchronize changes between machines.

Use `git push` to upload local commits and `git pull` to retrieve and integrate remote changes.

## Merge Conflicts

A merge conflict occurs when Git cannot automatically reconcile competing changes. Git marks the affected files so the developer can inspect and resolve the conflicting content.

How to Resolve a Merge Conflict

Run git status to identify files with conflicts.

Open each conflicted file and look for conflict markers: <<<<<<<, =======, and >>>>>>>.

Edit the file to keep the intended changes and remove all conflict markers.

Run git add <filename> to mark the file as resolved.

Run git status to verify that the file is no longer listed as conflicted.

Complete the merge with git commit if Git has not already completed it automatically.

Before resolving a conflict, understand the competing changes rather than blindly choosing one side. If necessary, consult your teammates or inspect the relevant commit history.

Preventing Merge Conflicts

Keep branches reasonably up to date, make focused commits, and communicate when multiple people are changing the same files. Frequently integrating changes can reduce the size of conflicts when they occur.