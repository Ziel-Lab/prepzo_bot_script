#!/usr/bin/env python3
"""
Script to check the status of Prepzo Bot deployment on EC2
"""

import argparse
import requests
import json
import os
import sys
from datetime import datetime

def check_instance_status(instance_ip, port=80):
    """Check if the instance is responding to HTTP requests"""
    try:
        url = f"http://{instance_ip}:{port}/health"
        print(f"Checking status at {url}...")
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is up and running! Status code: {response.status_code}")
            try:
                data = response.json()
                print(f"Server info: {json.dumps(data, indent=2)}")
                return True
            except:
                print("Server responded but did not return valid JSON")
                return True
        else:
            print(f"❌ Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error - server may not be running")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out - server may be overloaded")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {str(e)}")
        return False

def check_service_status(instance_ip, ssh_key, username="ec2-user"):
    """Check the status of the bot service via SSH"""
    try:
        import paramiko
        
        print(f"Connecting to {instance_ip} via SSH...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the server
        client.connect(instance_ip, username=username, key_filename=ssh_key)
        
        # Check systemd service status
        print("Checking bot service status...")
        stdin, stdout, stderr = client.exec_command('sudo systemctl status prepzo-bot')
        service_status = stdout.read().decode('utf-8')
        print("\n--- Service Status ---")
        print(service_status)
        print("---------------------\n")
        
        # Check logs
        print("Checking recent logs...")
        stdin, stdout, stderr = client.exec_command('sudo journalctl -u prepzo-bot -n 20')
        logs = stdout.read().decode('utf-8')
        print("\n--- Recent Logs ---")
        print(logs)
        print("-------------------\n")
        
        # Close the connection
        client.close()
        
        if "Active: active (running)" in service_status:
            print("✅ Prepzo Bot service is running")
            return True
        else:
            print("❌ Prepzo Bot service is not running")
            return False
    except ImportError:
        print("Could not import paramiko. Install it with: pip install paramiko")
        return False
    except Exception as e:
        print(f"❌ SSH connection error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check Prepzo Bot deployment status")
    parser.add_argument("--ip", required=True, help="EC2 instance public IP address")
    parser.add_argument("--ssh-key", help="Path to SSH key for instance access")
    parser.add_argument("--port", type=int, default=80, help="Server port (default: 80)")
    parser.add_argument("--username", default="ec2-user", help="SSH username (default: ec2-user)")
    
    args = parser.parse_args()
    
    print(f"Checking deployment status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Instance IP: {args.ip}")
    
    # First check HTTP status
    http_status = check_instance_status(args.ip, args.port)
    
    # Then check service status if SSH key is provided
    service_status = False
    if args.ssh_key:
        if os.path.exists(args.ssh_key):
            service_status = check_service_status(args.ip, args.ssh_key, args.username)
        else:
            print(f"❌ SSH key file not found: {args.ssh_key}")
    else:
        print("SSH key not provided, skipping service status check")
    
    # Overall status
    if http_status and (service_status or not args.ssh_key):
        print("\n✅ Deployment appears to be successful!")
        return 0
    else:
        print("\n❌ Deployment has issues that need to be addressed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 