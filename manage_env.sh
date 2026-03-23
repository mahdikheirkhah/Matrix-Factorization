#!/bin/bash

# Configuration
ENV_NAME="ex00"
REQ_FILE="requirements.txt"
PORT=8891

# --- Helper Functions ---

kill_jupyter() {
    PID=$(lsof -t -i:$PORT)
    if [ -n "$PID" ]; then
        echo "Stopping Jupyter server on port $PORT..."
        kill -9 $PID
    fi
}

cleanup_kernels() {
    echo "Scrubbing stale kernels..."
    jupyter kernelspec uninstall "$ENV_NAME" -f 2>/dev/null
    jupyter kernelspec list | grep -E "localhost|127.0.0.1" | awk '{print $1}' | xargs -I {} jupyter kernelspec uninstall {} -f 2>/dev/null
}

setup_vscode_settings() {
    local TOKEN=$1
    echo "Configuring VS Code workspace settings..."
    mkdir -p .vscode
    ABS_PATH=$(pwd)
    
    cat <<EOF > .vscode/settings.json
{
    "python.defaultInterpreterPath": "$ABS_PATH/$ENV_NAME/bin/python",
    "jupyter.jupyterServerType": "local",
    "jupyter.notebookEditor.defaultKernel": ".jupyter/kernels/$ENV_NAME",
    "python.terminal.activateEnvInSelectedTerminal": true,
    "jupyter.jupyterServerEndpoint": "http://localhost:$PORT/?token=$TOKEN"
}
EOF
}

ensure_jupyter_installed() {
    # Check using the venv's pip specifically
    if ! $ENV_NAME/bin/pip show notebook > /dev/null 2>&1; then
        echo "Jupyter Notebook not found. Installing..."
        $ENV_NAME/bin/pip install jupyter
        [[ ! -f "$REQ_FILE" ]] && touch "$REQ_FILE"
        grep -qi "jupyter" "$REQ_FILE" || echo "jupyter" >> "$REQ_FILE"
        grep -qi "notebook" "$REQ_FILE" || echo "notebook" >> "$REQ_FILE"
        grep -qi "ipykernel" "$REQ_FILE" || echo "ipykernel" >> "$REQ_FILE"
    fi
}

launch_terminal_and_run() {
    echo "Opening Jupyter in a new terminal window..."
    CMD="cd '$(pwd)' && source $ENV_NAME/bin/activate && jupyter notebook --port $PORT --no-browser"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e "tell application \"Terminal\" to do script \"$CMD\""
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        gnome-terminal -- bash -c "$CMD; exec bash" || xterm -e "$CMD"
    fi

    echo "Waiting for Jupyter to start..."
    sleep 6 
    
    TOKEN=$(jupyter notebook list 2>/dev/null | grep "$PORT" | sed -E 's/.*token=([^ ]+).*/\1/' | head -n 1)
    [[ -z "$TOKEN" ]] && TOKEN=$(jupyter server list 2>/dev/null | grep "$PORT" | sed -E 's/.*token=([^ ]+).*/\1/' | head -n 1)

    setup_vscode_settings "$TOKEN"
    echo "Token found: ${TOKEN:0:10}..."
}

create_and_init() {
    if [ -d "$ENV_NAME" ]; then
        echo "Environment exists. Activating it now..."
    else
        echo "Creating $ENV_NAME..."
        python3 -m venv $ENV_NAME
        [ ! -f .gitignore ] && touch .gitignore
        for entry in "$ENV_NAME" ".DS_Store"; do
            grep -q "^$entry$" .gitignore || echo "$entry" >> .gitignore
        done
    fi
    
    # ACTIVATE in current shell
    source $ENV_NAME/bin/activate
    
    # Use explicit venv pip to ensure it goes to the right place
    $ENV_NAME/bin/pip install --upgrade pip
    $ENV_NAME/bin/pip install numpy pandas matplotlib scikit_learn
    if [ -s "$REQ_FILE" ]; then
        $ENV_NAME/bin/pip install -r "$REQ_FILE"
    else
        echo "Message: $REQ_FILE is empty."
    fi

    ensure_jupyter_installed

    # Install/Update the kernel spec using the venv's python
    $ENV_NAME/bin/python -m ipykernel install --user --name="$ENV_NAME" --display-name="Python ($ENV_NAME)"
    
    kill_jupyter
    cleanup_kernels
    launch_terminal_and_run
    $ENV_NAME/bin/pip freeze > $REQ_FILE
    
    echo "✅ Setup finished."
    echo "⚠️ NOTE: To remain in the activated environment, you must run this script with 'source'."
}

# --- Flag Logic ---

if [[ $# -eq 0 ]]; then
    create_and_init
else
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -c) python3 -m venv "$ENV_NAME" ;;
            -a) source $ENV_NAME/bin/activate ;;
            -i) 
                shift; PACKAGES=""
                while [[ "$1" != -* && -n "$1" ]]; do PACKAGES="$PACKAGES $1"; shift; done
                # Use absolute path to the venv's pip to prevent global installs
                $ENV_NAME/bin/pip install $PACKAGES && $ENV_NAME/bin/pip freeze > "$REQ_FILE"
                continue ;;
            --run) launch_terminal_and_run ;;
            -d) [[ -n "$VIRTUAL_ENV" ]] && deactivate && echo "Deactivated." ;;
            --delete) 
                kill_jupyter; cleanup_kernels
                [[ -n "$VIRTUAL_ENV" ]] && deactivate
                rm -rf "$ENV_NAME" .vscode/settings.json
                echo "Deleted everything."
                ;;
            *) 
                echo "Unknown flag: $1"
                # Use return instead of exit to prevent closing the terminal when sourced
                return 1 2>/dev/null || exit 1 
                ;;
        esac
        shift
    done
fi