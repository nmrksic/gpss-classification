import subprocess
import shlex
import os.path

config_params = []

username = raw_input("What is your user name: ")
config_params.append(('USERNAME', username))
print config_params

location = raw_input("What is your location (home or local): ")
config_params.append(('LOCATION', location))
print config_params

if location == 'local':
    # Creating local 2 fear key pair
    command = 'ssh-keygen -t rsa -f ' + os.path.expanduser('~/.ssh/%slocal2fear' % username)
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
	# Create ~/.ssh directory if necessary
    command = 'ssh %s@fear "[ -d ~/.ssh ] || mkdir ~/.ssh"' % username
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    # Sending public key to fear
    command = 'scp ' + os.path.expanduser('~/.ssh/%slocal2fear.pub' % username) + ' %s@fear:~/.ssh/' % username
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    # Putting key into authorized keys file
    command = 'ssh %s@fear "cat >> ~/.ssh/authorized_keys" < ' % username + '~/.ssh/%slocal2fear.pub' % username
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    #### TODO - record the locations of the key files
    # Adding key to SSH agent
    # command = 'ssh-add ~/.ssh/%slocal2fear' % username #### WARNING - assumes SSH authorization agent - doesn't work anyway - -i is the nicer thing to do
    # print 'COMMAND = %s' % command
    # subprocess.call(shlex.split(command))
    # Creating key pair on fear
    command = 'ssh -i ' + os.path.expanduser('~/.ssh/%slocal2fear.pub' % username) +  ' %s@fear ' % username + '"ssh-keygen -t rsa -f ~/.ssh/%sfear2local"' % username
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    # Retrieving public key from fear
    command = 'scp -i ' + os.path.expanduser('~/.ssh/%slocal2fear.pub' % username) +  ' %s@fear:~/.ssh/%sfear2local.pub ' % (username, username) + os.path.expanduser('~/.ssh/') 
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    # Putting key into authorized keys file
    command = 'cat ' + os.path.expanduser('~/.ssh/%sfear2local.pub' % username) + ' >> ' + os.path.expanduser('~/.ssh/authorized_keys') 
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    # Creating temp directory on fear
    pass
    # Creating python directory on fear
    pass
    # Creating matlab directory on fear
    pass
    # Create GPML directory - or part of a separate installer?
    #### TODO - don't forget dependencies - pysftp
    
elif location == 'home':
    # Creating home 2 gate key pair
    command = 'ssh-keygen -t rsa -f ' + os.path.expanduser('~/.ssh/%shome2gate' % username)
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    # Sending public key to gate
    command = 'scp ' + os.path.expanduser('~/.ssh/%shome2gate.pub' % username) + ' %s@gate.eng.cam.ac.uk:~/.ssh/' % username
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    # Putting key into authorized keys file
    command = 'ssh %s@gate.eng.cam.ac.uk "cat >> ~/.ssh/authorized_keys" < ' % username + '~/.ssh/%shome2gate.pub' % username
    print 'COMMAND = %s' % command
    subprocess.call(shlex.split(command))
    #### TODO - record the locations of the key files and create yet more key files
else:
    raise Exception('I am a very brittle program')
