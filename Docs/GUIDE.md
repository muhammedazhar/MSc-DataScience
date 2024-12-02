# Git LFS Troubleshooting Guide

## Common Issues and Solutions

### 1. File Size Exceeds GitHub Limit

**Error:**

```md
remote: error: File X is 206.71 MB; this exceeds GitHub's file size limit of 100.00 MB
```

**Solution:**

```bash
# Remove file from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/large/file" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Setup LFS tracking
git lfs track "*.pdf"  # or relevant file extension
git add .gitattributes
git commit -m "Setup Git LFS tracking"

# Add file back and push
git add path/to/large/file
git commit -m "Add large file via Git LFS"
git push origin master --force
```

## 2. Permission Denied Errors

**Error:**

```md
Error cleaning Git LFS object: permission denied
```

**Solution:**

```bash
# Fix permissions
sudo chown -R $(whoami) .git/lfs
chmod -R 755 .git/lfs

# Reinstall LFS
git lfs uninstall
git lfs install
```

## 3. System-wide Installation Issues

**Error:**

```md
warning: current user is not root/admin, system install is likely to fail
```

**Solution:**

```bash
# Option 1: Install system-wide (requires admin)
sudo git lfs install --system

# Option 2: Install for current user only
git lfs install
```

## Best Practices

1. Setup Git LFS before adding large files
2. Track specific file types: `git lfs track "*.pdf" "*.psd" "*.zip"`
3. Always commit `.gitattributes` first
4. Verify tracking: `git lfs status`

## Quick Reference

```bash
# Initialize Git LFS
git lfs install

# Track file types
git lfs track "*.pdf"

# Verify tracking
git lfs ls-files

# Pull LFS objects
git lfs pull

# Clean LFS cache
git lfs prune
```
