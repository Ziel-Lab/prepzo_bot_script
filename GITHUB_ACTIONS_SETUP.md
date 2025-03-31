# GitHub Actions CI/CD Setup for Prepzo Bot

This guide explains how to set up continuous integration and deployment (CI/CD) for the Prepzo bot using GitHub Actions and AWS.

## Prerequisites

1. An AWS account
2. GitHub repository with your Prepzo bot code
3. IAM user for GitHub deployment named `github_deployment`

## Setting Up AWS IAM for GitHub Actions

### 1. Create an IAM User for GitHub Actions

1. Open the AWS Management Console and navigate to IAM
2. Select "Users" and click "Add users"
3. Name the user `github_deployment`
4. Select "Access key - Programmatic access" as the access type
5. Click "Next: Permissions"

### 2. Attach Policies to the User

Attach the following policies to the user:
- `AmazonEC2FullAccess`
- `AmazonVPCFullAccess`
- `IAMFullAccess`
- `CloudFormationFullAccess`

For production, you should consider creating a more restricted custom policy, but these permissions are sufficient for development and testing.

### 3. Create Access Keys

1. Complete the user creation process
2. Save the Access key ID and Secret access key - you'll need these for GitHub Secrets

## GitHub Repository Secrets

Add the following secrets to your GitHub repository:

### AWS Access Credentials
1. `AWS_ACCESS_KEY_ID`: Access key for the `github_deployment` IAM user
2. `AWS_SECRET_ACCESS_KEY`: Secret key for the `github_deployment` IAM user
3. `AWS_REGION`: Set to `eu-north-1` for Stockholm
4. `EC2_EC2_KEY_NAME`: The name of your EC2 key pair for SSH access

### Application Secrets
5. `SUPABASE_URL`: Your Supabase project URL
6. `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key
7. `OPENAI_API_KEY`: Your OpenAI API key
8. `DEEPGRAM_API_KEY`: Your Deepgram API key
9. `LIVEKIT_URL`: Your LiveKit server URL
10. `LIVEKIT_API_KEY`: Your LiveKit API key
11. `CARTESIA_API_KEY`: Your Cartesia API key
12. `ELEVENLABS_API_KEY`: Your Eleven Labs API key

## How It Works

1. When you push to the `main` branch, GitHub Actions will automatically trigger a deployment.
2. The workflow will:
   - Authenticate with AWS using the access key and secret key
   - Set up Node.js and Python
   - Initialize a new AWS CDK project in TypeScript
   - Create an EC2 stack that provisions:
     - A VPC with public subnets
     - Security group with SSH, HTTP, HTTPS, and custom port access
     - An EC2 instance running Amazon Linux 2023
     - A systemd service to run your Prepzo bot
   - Deploy the stack to AWS
   - Output the public IP of your EC2 instance

## Manual Deployment

You can also trigger a manual deployment by:
1. Go to your GitHub repository
2. Click on the "Actions" tab
3. Select the "Deploy to EC2" workflow
4. Click "Run workflow"
5. Select the branch to deploy from
6. Click "Run workflow" again

## Troubleshooting

If deployment fails, check:
1. GitHub repository secrets are correctly set
2. IAM user has the necessary permissions
3. GitHub Actions logs for specific error messages 