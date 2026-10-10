# GitHub Repository Integration & Sync Guide

Your local project workspace is already initialized with Git and connected to your GitHub repository!

---

## 1. Current Remote Configuration

To verify your currently configured GitHub remote, run in your terminal:
```bash
git remote -v
```

Currently, your remote is mapped to:
- **origin (fetch):** `https://github.com/shaheshaan2266-engg/ai-entertainment-recommender.git`
- **origin (push):** `https://github.com/shaheshaan2266-engg/ai-entertainment-recommender.git`

---

## 2. Pushing Updates to GitHub

To commit and push all new datasets, recommender models, chaining engine, and web UI to your GitHub repository:

```bash
# 1. Stage all changes
git add .

# 2. Commit the changes
git commit -m "Add Indian Cinema & Music AI Recommender, Chaining Engine, and Fluid Web UI"

# 3. Push to your main branch
git push origin main
```

> **Note on Authentication**: If Git prompts you for credentials:
> - **Username**: your GitHub username
> - **Password**: your **GitHub Personal Access Token (Classic or Fine-grained)** with `repo` scope permissions (GitHub does not accept standard account passwords over HTTPS).

---

## 3. Changing to a Different GitHub Repository

If you want to link a new or different GitHub repository:

```bash
# Set new repository URL
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_NEW_REPO_NAME.git

# Verify the updated remote
git remote -v

# Push to the new repository
git push -u origin main
```

---

## 4. SSH Authentication (Optional Alternative)

If you have SSH keys configured on your machine:
```bash
git remote set-url origin git@github.com:shaheshaan2266-engg/ai-entertainment-recommender.git
git push origin main
```
