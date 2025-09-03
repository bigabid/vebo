# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/d647a14c-4bf1-4aca-82dd-bfca0318e417

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/d647a14c-4bf1-4aca-82dd-bfca0318e417) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## Running the Athena Insights flow locally

Set environment variables and run three processes: UI, Node server, and Python insights API.

1) Node server (Athena metadata + execute)

Required env:

- `AWS_REGION` or `AWS_DEFAULT_REGION`
- `AWS_PROFILE` (optional)
- `ATHENA_OUTPUT_S3` (e.g., `s3://my-bucket/athena-results/`)
- `ATHENA_WORKGROUP` (optional)

Start:

```bash
cd website
node server/index.js
```

2) Python insights API (poll Athena, run profiler)

Install deps and run:

```bash
cd ../python
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

3) UI

```bash
cd ../website
VITE_INSIGHTS_API=http://localhost:8000 npm run dev
```

Flow:

- UI calls `POST /api/execute` on Node to start Athena query and receive `executionId`.
- UI calls Python `POST /insights/start` with the `executionId`.
- UI polls Python `GET /insights/status?jobId=...` until status is `complete` and then renders insights.

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/d647a14c-4bf1-4aca-82dd-bfca0318e417) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/tips-tricks/custom-domain#step-by-step-guide)
