# Environment Variables Guide

This guide provides instructions on how to set, unset, and manage environment variables across different operating systems, with a focus on handling sensitive information such as API keys securely. Environment variables are key-value pairs that can influence the behavior of software on your computer. While they offer a convenient way to configure software behaviors without hard-coding sensitive information, it's crucial to manage them with care. This includes avoiding setting sensitive data globally on shared or public computers and ensuring environment variables are only accessible to users that require them.


## Setting Environment Variables Temporarily (Per Terminal Session)

Guide for setting/unsetting an environment variable for your current terminal session.
After you close the terminal, the environment variable will be erased.

Use cases: for testing new configurations or when working with temporary credentials.

### Linux and macOS

To set variable:

```bash
export VARIABLE_NAME="value"
```

To remove it within the same session:

```bash
unset VARIABLE_NAME
```

### Windows

To set in Command Prompt (cmd.exe):

```cmd
set VARIABLE_NAME="value"
```

To unset in Command Prompt (cmd.exe):

```cmd
set VARIABLE_NAME=
```

To set in PowerShell:

```powershell
$env:VARIABLE_NAME="value"
```

To unset in PowerShell:

```powershell
Remove-Item Env:\VARIABLE_NAME
```

## Permanently (Per User)

Guide on how to set an environment variable so it persists across all terminal sessions of a current user.

Use cases: you have a personal computer or a dedicated workstation where you are the primary user, you regularly use specific software tools or commands that rely on environment variables for configuration.

### Linux and macOS

You need to add it to a special file that runs whenever you open a terminal. For users who haven't customized their shell environment, here are the paths of the default configuration files:

- Linux: `~/.bashrc`
- macOS: `~/.bash_profile` (on versions before Catalina), `~/.zshrc` (Catalina and later)

To add new environment variable to the configuration either run command:

```bash
echo 'export VARIABLE_NAME="value"' >> configuration_file_path
```

Or go to the end of the file and manually add your variable (please be cautious with syntax to avoid mistakes):

`export VARIABLE_NAME="value"`

In both cases, to apply the changes, reload terminal or run the command:

```bash
source configuration_file_path
```

### Windows

#### Option 1: Using UI

- Right-click on "This PC" or "Computer" on the desktop or in File Explorer
- Click "Properties" -> "Advanced system settings" -> "Environment Variables"
- Add or remove variables under **User** variables.

#### Option 2: Using CLI

To set in Command Prompt (cmd.exe):

```cmd
setx VARIABLE_NAME=value
```

To unset in Command Prompt (cmd.exe):

```cmd
setx VARIABLE_NAME=
```

Note: unsetting in cmd will not actually unset the environment variable, instead, it will set the variable to an empty string.
Use any of other proposed options for the full unset.

Note: `setx` expects quotes **only** around the values containing spaces. Otherwise, quotes will be considered as a part of value.


To set in PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("VARIABLE_NAME", "value", [EnvironmentVariableTarget]::User)
```

To unset in PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("VARIABLE_NAME", $null, [EnvironmentVariableTarget]::User)
```

In all cases, reload terminal for changes to make effect.


## Globally (System-Wide)

Guide on how to set a variable globally, system-wide, for all sessions and all users at login.

Use cases: environment variables have to be configured for all users on a shared server or workstation.

Please be careful with this option and don't use it if not actually needed!

Caution is advised since improper use can affect system behavior!

### Linux and macOS

You need to add variable to the file that is read by all sessions and all users at login.

- Linux: `/etc/environment`
- macOS: `/etc/profile`

The command for setting environment variable globbaly (requires superuser permissions):

```bash
sudo sh -c 'echo "VARIABLE_NAME=value" >> configuration_file_path'
```

### Windows

Doing this requires administrative privileges.

#### Option 1: Using UI

- Right-click on "This PC" or "Computer" on the desktop or in File Explorer
- Click "Properties" -> "Advanced system settings" -> "Environment Variables"
- Add or remove variables under **System** variables.

#### Option 2: Using CLI

To set in Command Prompt (cmd.exe):

```cmd
setx VARIABLE_NAME value /M
```

To unset in Command Prompt (cmd.exe):

```cmd
setx VARIABLE_NAME /M
```

Note: unsetting in cmd will not actually unset the environment variable, instead, it will set the variable to an empty string.
Use any of other proposed options for the full unset.

Note: `setx` expects quotes **only** around the values containing spaces. Otherwise, quotes will be considered as a part of value.


To set in PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("VARIABLE_NAME", "value", [EnvironmentVariableTarget]::Machine)
```

To unset in PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("VARIABLE_NAME", $null, [EnvironmentVariableTarget]::Machine)
```

In all cases, reload terminal for changes to make effect.
