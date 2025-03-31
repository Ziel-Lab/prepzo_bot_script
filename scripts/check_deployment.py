#!/usr/bin/env python
'''
Check if the deployment has completed successfully
'''

import argparse
import json
import requests
import time
import sys
from datetime import datetime


def check_endpoint(ip, endpoint="/health", port=8080, max_retries=10, retry_interval=5):
    """
    Check if the application's health endpoint is responding
    """
    url = f"http://{ip}:{port}{endpoint}"
    print(f"Checking health at {url}")
    
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}/{max_retries}...")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    print(f"✅ Health check succeeded!")
                    print(f"Status: {health_data.get('status', 'unknown')}")
                    print(f"Service: {health_data.get('service', 'unknown')}")
                    
                    version_info = health_data.get('version', {})
                    if isinstance(version_info, dict):
                        print(f"Version: {version_info.get('version', 'unknown')}")
                        print(f"Build date: {version_info.get('build_date', 'unknown')}")
                        print(f"Git commit: {version_info.get('git_commit', 'unknown')}")
                    else:
                        print(f"Version: {version_info}")
                    
                    return True
                except (json.JSONDecodeError, ValueError):
                    print(f"Warning: Received non-JSON response")
                    print(response.text[:100])  # Show first 100 chars of response
                    return True
            else:
                print(f"Got HTTP {response.status_code} response")
                
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            
        if i < max_retries - 1:
            print(f"Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
    
    return False


def check_aws_deployment_status(application_name, deployment_group):
    """
    Check the status of the most recent CodeDeploy deployment
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        client = boto3.client('codedeploy')
        
        # Get the most recent deployment
        try:
            response = client.list_deployments(
                applicationName=application_name,
                deploymentGroupName=deployment_group,
                includeOnlyStatuses=['Created', 'Queued', 'InProgress', 'Succeeded', 'Failed', 'Stopped', 'Ready'],
                limit=1
            )
            
            if not response.get('deployments'):
                print(f"No deployments found for {application_name}/{deployment_group}")
                return False
            
            deployment_id = response['deployments'][0]
            
            # Get deployment info
            deployment = client.get_deployment(deploymentId=deployment_id)
            status = deployment['deploymentInfo']['status']
            create_time = deployment['deploymentInfo']['createTime']
            
            # Format time nicely
            formatted_time = datetime.fromtimestamp(create_time.timestamp()).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Latest deployment ({deployment_id}):")
            print(f"  Status: {status}")
            print(f"  Created: {formatted_time}")
            
            if status == 'Succeeded':
                print("✅ Deployment completed successfully!")
                return True
            elif status in ['Failed', 'Stopped']:
                print("❌ Deployment failed or was stopped")
                # Get error information if available
                if 'errorInformation' in deployment['deploymentInfo']:
                    print(f"  Error code: {deployment['deploymentInfo']['errorInformation'].get('code', 'unknown')}")
                    print(f"  Error message: {deployment['deploymentInfo']['errorInformation'].get('message', 'unknown')}")
                return False
            else:
                print(f"Deployment is {status}")
                return None  # Still in progress
            
        except ClientError as e:
            print(f"Error getting deployment status: {e}")
            return False
            
    except ImportError:
        print("AWS boto3 library not installed. Can't check CodeDeploy status.")
        return None


def main():
    parser = argparse.ArgumentParser(description='Check deployment status')
    parser.add_argument('--ip', required=True, help='IP address of the deployed instance')
    parser.add_argument('--port', type=int, default=8080, help='Port for health check (default: 8080)')
    parser.add_argument('--app', default='PrepzoBotApplication', help='CodeDeploy application name')
    parser.add_argument('--group', default='PrepzoBotDeploymentGroup', help='CodeDeploy deployment group')
    parser.add_argument('--retries', type=int, default=10, help='Maximum number of retries')
    parser.add_argument('--interval', type=int, default=5, help='Seconds between retries')
    args = parser.parse_args()
    
    # First check AWS deployment status if possible
    aws_status = check_aws_deployment_status(args.app, args.group)
    
    if aws_status is False:
        print("AWS deployment check failed. Still checking service health...")
    
    # Then check the actual endpoint
    health_status = check_endpoint(
        args.ip, 
        port=args.port,
        max_retries=args.retries,
        retry_interval=args.interval
    )
    
    if health_status:
        print("\nService is up and running correctly!")
        return 0
    else:
        print("\nService health check failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 