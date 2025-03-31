#!/bin/bash
# Enhanced deployment script for Prepzo Bot with AWS CodeDeploy integration

set -e  # Exit on error

# Parse arguments
DEPLOY_TYPE="standard"
VERSION=""
FORCE_DEPLOY=false

print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --version VERSION    Update version.py to specified version"
  echo "  --check              Just check deployment status without deploying"
  echo "  --force              Force deploy even with uncommitted changes"
  echo "  --help               Show this help message"
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --version) VERSION="$2"; shift ;;
    --check) DEPLOY_TYPE="check"; shift ;;
    --force) FORCE_DEPLOY=true; shift ;;
    --help) print_usage; exit 0 ;;
    *) echo "Unknown parameter: $1"; print_usage; exit 1 ;;
  esac
  shift
done

# Get repository info
REPO_URL=$(git config --get remote.origin.url)
REPO_NAME=$(echo $REPO_URL | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
GITHUB_ACTIONS_URL="https://github.com/$REPO_NAME/actions"

# Update version.py with current date
today=$(date +"%Y-%m-%d")
if [ -f "version.py" ]; then
  echo "Updating build date to $today"
  sed -i "s/BUILD_DATE = \".*\"/BUILD_DATE = \"$today\"/" version.py
fi

# Get the current git commit
git_commit=$(git rev-parse HEAD)
echo "Current commit: $git_commit"

# Check if there are uncommitted changes
if [[ -n $(git status -s) ]]; then
  if [[ "$FORCE_DEPLOY" == true ]]; then
    echo "Warning: You have uncommitted changes, but proceeding because --force was specified."
  else
    echo "Error: You have uncommitted changes. Commit before deploying or use --force to override."
    exit 1
  fi
fi

# If we're just checking status, run check_deployment.py
if [[ "$DEPLOY_TYPE" == "check" ]]; then
  echo "Checking deployment status..."
  
  # Get instance IP
  INSTANCE_IP=$(aws ec2 describe-instances \
    --filters "Name=tag:aws:cloudformation:stack-name,Values=PrepzoBotStack" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
  
  if [ -z "$INSTANCE_IP" ] || [ "$INSTANCE_IP" == "None" ]; then
    echo "Error: Could not find instance IP"
    exit 1
  fi
  
  echo "Instance IP: $INSTANCE_IP"
  
  if [ -f "scripts/check_deployment.py" ]; then
    python scripts/check_deployment.py --ip $INSTANCE_IP
    exit $?
  else
    echo "Error: check_deployment.py not found"
    exit 1
  fi
fi

# Update version if specified
if [[ ! -z "$VERSION" ]]; then
  echo "Updating version to $VERSION"
  sed -i "s/VERSION = \".*\"/VERSION = \"$VERSION\"/" version.py
  git add version.py
  git commit -m "Update version to $VERSION"
fi

# Create deployment package locally
echo "Creating local deployment package..."

# Create temporary directory for deployment package
TEMP_DIR=$(mktemp -d)
mkdir -p $TEMP_DIR/scripts

# Create appspec.yml
cat > $TEMP_DIR/appspec.yml << 'EOF'
version: 0.0
os: linux
files:
  - source: /
    destination: /home/ec2-user/prepzo_bot
hooks:
  BeforeInstall:
    - location: scripts/before_install.sh
      timeout: 300
      runas: root
  AfterInstall:
    - location: scripts/after_install.sh
      timeout: 300
      runas: root
  ApplicationStart:
    - location: scripts/start_application.sh
      timeout: 300
      runas: root
  ValidateService:
    - location: scripts/validate_service.sh
      timeout: 300
      runas: root
EOF

# Create deployment scripts
cat > $TEMP_DIR/scripts/before_install.sh << 'EOF'
#!/bin/bash
echo "Before installation steps starting at $(date)"

# Stop services if they exist
if systemctl is-active --quiet prepzo-bot; then
    systemctl stop prepzo-bot
fi
if systemctl is-active --quiet prepzo-health; then
    systemctl stop prepzo-health
fi

# Create directories if they don't exist
mkdir -p /home/ec2-user/.env
mkdir -p /home/ec2-user/prepzo_bot
EOF

cat > $TEMP_DIR/scripts/after_install.sh << 'EOF'
#!/bin/bash
echo "After installation steps starting at $(date)"

# Set ownership
chown -R ec2-user:ec2-user /home/ec2-user/prepzo_bot

# Set up Python environment
cd /home/ec2-user/prepzo_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt || echo "No requirements.txt found"
pip install requests flask boto3

# Load environment variables from SSM Parameter Store
echo "Loading environment variables from SSM Parameter Store"
cat > /home/ec2-user/.env/prepzo_bot.env << 'ENVEOF'
SUPABASE_URL=$(aws ssm get-parameter --name "/prepzo-bot/SUPABASE_URL" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
SUPABASE_SERVICE_ROLE_KEY=$(aws ssm get-parameter --name "/prepzo-bot/SUPABASE_SERVICE_ROLE_KEY" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
OPENAI_API_KEY=$(aws ssm get-parameter --name "/prepzo-bot/OPENAI_API_KEY" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
DEEPGRAM_API_KEY=$(aws ssm get-parameter --name "/prepzo-bot/DEEPGRAM_API_KEY" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
LIVEKIT_URL=$(aws ssm get-parameter --name "/prepzo-bot/LIVEKIT_URL" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
LIVEKIT_API_KEY=$(aws ssm get-parameter --name "/prepzo-bot/LIVEKIT_API_KEY" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
CARTESIA_API_KEY=$(aws ssm get-parameter --name "/prepzo-bot/CARTESIA_API_KEY" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
ELEVENLABS_API_KEY=$(aws ssm get-parameter --name "/prepzo-bot/ELEVENLABS_API_KEY" --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")
HEALTH_CHECK_PORT=8080
ENVIRONMENT=production
GIT_COMMIT=$(cat /home/ec2-user/prepzo_bot/GIT_COMMIT || echo "unknown")
ENVEOF

# Create health server file
cat > /home/ec2-user/prepzo_bot/health_server.py << 'EOF'
import flask
import os
import json
import sys

sys.path.append('/home/ec2-user/prepzo_bot')
try:
    from version import get_version_info
    version_info = get_version_info()
except ImportError:
    version_info = {"version": "unknown", "build_date": "unknown", "git_commit": "unknown"}

app = flask.Flask(__name__)

@app.route("/health")
def health():
    return {"status": "ok", "service": "prepzo-bot", "version": version_info}

if __name__ == "__main__":
    port = int(os.environ.get("HEALTH_CHECK_PORT", 8080))
    app.run(host="0.0.0.0", port=port)
EOF

# Create service files
cat > /etc/systemd/system/prepzo-health.service << 'EOF'
[Unit]
Description=Prepzo Health Check Service
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/prepzo_bot
EnvironmentFile=/home/ec2-user/.env/prepzo_bot.env
ExecStart=/home/ec2-user/prepzo_bot/venv/bin/python health_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/prepzo-bot.service << 'EOF'
[Unit]
Description=Prepzo Bot Service
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/prepzo_bot
EnvironmentFile=/home/ec2-user/.env/prepzo_bot.env
ExecStart=/home/ec2-user/prepzo_bot/venv/bin/python main.py start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd configuration
systemctl daemon-reload
EOF

cat > $TEMP_DIR/scripts/start_application.sh << 'EOF'
#!/bin/bash
echo "Starting application at $(date)"

# Start services
systemctl enable prepzo-health
systemctl start prepzo-health
systemctl enable prepzo-bot
systemctl start prepzo-bot || echo "Main service failed to start, but health endpoint should be running"

# Set up port forwarding
iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080 || echo "Failed to set up port forwarding"
EOF

cat > $TEMP_DIR/scripts/validate_service.sh << 'EOF'
#!/bin/bash
echo "Validating service at $(date)"

# Check if the health endpoint is running
HEALTH_CHECK_PORT=${HEALTH_CHECK_PORT:-8080}
MAX_RETRIES=30
RETRY_INTERVAL=2

for i in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $i of $MAX_RETRIES..."
    
    # Try the health endpoint
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$HEALTH_CHECK_PORT/health || echo "error")
    
    if [ "$RESPONSE" == "200" ]; then
        echo "✅ Service is up and running!"
        exit 0
    fi
    
    echo "Service not ready yet (response: $RESPONSE), waiting $RETRY_INTERVAL seconds..."
    sleep $RETRY_INTERVAL
done

echo "❌ Service validation failed after $MAX_RETRIES attempts"
exit 1
EOF

# Make scripts executable
chmod +x $TEMP_DIR/scripts/*.sh

# Save current git commit
echo "$git_commit" > $TEMP_DIR/GIT_COMMIT

# Copy application files to deployment package
rsync -av --exclude={.git,.github,node_modules,infrastructure,deployment,.env} . $TEMP_DIR/ || {
  echo "rsync failed, falling back to manual copy"
  find . -name "*.py" -type f -exec cp --parents {} $TEMP_DIR/ \;
  
  if [ -f "requirements.txt" ]; then
    cp requirements.txt $TEMP_DIR/
  fi
}

# Create ZIP archive
(cd $TEMP_DIR && zip -r ../prepzo-bot-deployment.zip .)
echo "Deployment package created: prepzo-bot-deployment.zip"

# If AWS CLI is installed, offer to deploy directly
if command -v aws &> /dev/null; then
  read -p "Do you want to deploy using AWS CodeDeploy? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    # Check if S3 bucket exists
    BUCKET_NAME="prepzo-bot-deployment-$(echo $REPO_NAME | tr '/' '-')"
    BUCKET_EXISTS=$(aws s3api list-buckets --query "Buckets[?Name=='${BUCKET_NAME}'].Name" --output text)
    
    if [ -z "$BUCKET_EXISTS" ]; then
      echo "Creating S3 bucket: ${BUCKET_NAME}"
      aws s3api create-bucket --bucket ${BUCKET_NAME} --create-bucket-configuration LocationConstraint=eu-north-1
      aws s3api put-bucket-encryption --bucket ${BUCKET_NAME} --server-side-encryption-configuration '{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}'
    fi
    
    # Upload to S3
    S3_LOCATION="s3://${BUCKET_NAME}/${git_commit}.zip"
    echo "Uploading deployment package to ${S3_LOCATION}..."
    aws s3 cp prepzo-bot-deployment.zip ${S3_LOCATION}
    
    # Check if CodeDeploy application exists
    APP_NAME="PrepzoBotApplication"
    APP_EXISTS=$(aws deploy list-applications --query "applications[?contains(@,'${APP_NAME}')]" --output text)
    
    if [ -z "$APP_EXISTS" ]; then
      echo "Creating CodeDeploy application: ${APP_NAME}"
      aws deploy create-application --application-name ${APP_NAME}
    fi
    
    # Check if deployment group exists
    DEPLOY_GROUP="PrepzoBotDeploymentGroup"
    DEPLOY_GROUP_EXISTS=$(aws deploy list-deployment-groups --application-name ${APP_NAME} --query "deploymentGroups[?contains(@,'${DEPLOY_GROUP}')]" --output text 2>/dev/null || echo "")
    
    if [ -z "$DEPLOY_GROUP_EXISTS" ]; then
      echo "Please create a deployment group in the AWS console or use GitHub Actions for deployment"
    else
      # Create deployment
      echo "Creating deployment with CodeDeploy..."
      DEPLOYMENT_ID=$(aws deploy create-deployment \
        --application-name ${APP_NAME} \
        --deployment-group-name ${DEPLOY_GROUP} \
        --s3-location bucket=${BUCKET_NAME},bundleType=zip,key=${git_commit}.zip \
        --file-exists-behavior OVERWRITE \
        --query "deploymentId" --output text)
      
      echo "Deployment started with ID: ${DEPLOYMENT_ID}"
      echo "You can check the status in the AWS Console"
      
      # Poll deployment status
      echo "Waiting for deployment to complete..."
      MAX_RETRIES=30
      RETRY_INTERVAL=10
      
      for i in $(seq 1 $MAX_RETRIES); do
        echo "Checking deployment status (attempt $i of $MAX_RETRIES)..."
        DEPLOYMENT_STATUS=$(aws deploy get-deployment --deployment-id ${DEPLOYMENT_ID} --query "deploymentInfo.status" --output text)
        
        echo "Current status: ${DEPLOYMENT_STATUS}"
        
        if [ "$DEPLOYMENT_STATUS" == "Succeeded" ]; then
          echo "✅ Deployment successful!"
          break
        elif [ "$DEPLOYMENT_STATUS" == "Failed" ] || [ "$DEPLOYMENT_STATUS" == "Stopped" ]; then
          echo "❌ Deployment failed or was stopped"
          aws deploy get-deployment --deployment-id ${DEPLOYMENT_ID}
          break
        fi
        
        if [ $i -eq $MAX_RETRIES ]; then
          echo "Deployment still in progress. Please check AWS Console for further updates."
          break
        fi
        
        echo "Waiting ${RETRY_INTERVAL} seconds for next check..."
        sleep ${RETRY_INTERVAL}
      done
    fi
  else
    echo "Manual deployment with AWS CLI skipped."
  fi
fi

# Push to GitHub to trigger GitHub Actions
read -p "Do you want to push to GitHub to trigger GitHub Actions? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Pushing to GitHub..."
  git push origin main

  echo "Deployment initiated. Check GitHub Actions for progress."
  echo "GitHub Actions URL: $GITHUB_ACTIONS_URL"
fi

# Clean up
rm -rf $TEMP_DIR
echo "Deployment script completed successfully." 