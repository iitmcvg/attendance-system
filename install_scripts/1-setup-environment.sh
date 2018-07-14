# Set up zsh + zim
sh -c "$(wget https://gist.githubusercontent.com/rsnk96/87229bd910e01f2ee7c35f96d7cb2f6c/raw/f068812ebd711ed01ebc4c128c8624730ab0dc81/build-zsh.sh -O -)"

# Uncomment for zim
#git clone --recursive https://github.com/Eriner/zim.git ${ZDOTDIR:-${HOME}}/.zim
#ln -s ~/.zim/templates/zimrc ~/.zimrc
#ln -s ~/.zim/templates/zlogin ~/.zlogin
#ln -s ~/.zim/templates/zshrc ~/.zshrc

# Uncomment for oh my zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

sudo apt-get -y install tmux
sudo apt-get -y install npm

# Install axel, a download accelerator
if ! test -d "axel"; then
    git clone https://github.com/axel-download-accelerator/axel.git
else
    (
        cd axel || exit
        git pull
    )
fi

sudo apt-get install autopoint openssl libssl-dev -y
cd ~/HV/axel
./autogen.sh
./configure
make
sudo make install
cd ../

# Conda setups
continuum_website=https://repo.continuum.io/archive/

# Stepwise filtering of the html at $continuum_website
# Get the topmost line that matches our requirements, extract the file name.

latest_anaconda_steup=$(wget -q -O - $continuum_website index.html | grep "Anaconda3-" | grep "Linux" | grep "86_64" | head -n 1 | cut -d \" -f 2)

#axel -o ./anacondaInstallScript.sh "$continuum_website$latest_anaconda_steup"
#sudo mkdir /opt/anaconda3 && sudo chmod ugo+w /opt/anaconda3

wget -O install_scripts/anacondaInstallScript.sh "$continuum_website$latest_anaconda_steup"
bash install_scripts/anacondaInstallScript.sh -b -p $HOME/anaconda3

touch ~/.bash_aliases
echo "Adding aliases to ~/.bash_aliases"
{
    echo "alias jn=\"jupyter notebook\""
    echo "alias maxvol=\"pactl set-sink-volume @DEFAULT_SINK@ 150%\""
    echo "alias download=\"wget --random-wait -r -p --no-parent -e robots=off -U mozilla\""
    echo "alias server=\"ifconfig | grep inet\\ addr && python3 -m http.server\""
    echo "weather() {curl wttr.in/\"\$1\";}"
    echo "alias gpom=\"git push origin master\""
    echo "alias update=\"sudo apt-get update && sudo apt-get dist-upgrade && sudo apt-get autoremove -y\""
    echo "alias tmux=\"tmux -u new-session \\; \\
            neww \\; \\
              send-keys 'htop' C-m \\; \\
              split-window -h \\; \\
              send-keys 'nvidia-smi -l 1' C-m \\; \\
              rename-window 'performance' \\; \\
            select-window -l\""

    echo "export PATH="$HOME/anaconda3/bin:$PATH""



} >> ~/.bash_aliases

source $HOME/anaconda3/bin/activate

echo "Adding anaconda to path variables"
{
    echo ""
    echo "export OLDPATH=\$PATH"
    echo "export PATH=$HOME/anaconda3/bin:\$PATH"

    echo "if [ -f ~/.bash_aliases ]; then"
    echo "  source ~/.bash_aliases"
    echo "fi"
} >> ~/.zshrc

sudo apt-get install -y sshfs

# Remote atom rmate
curl -o /usr/local/bin/rmate https://raw.githubusercontent.com/aurora/rmate/master/rmate
sudo chmod +x /usr/local/bin/rmate

# Change default shell to zsh
command -v zsh | sudo tee -a /etc/shells
sudo chsh -s "$(command -v zsh)" "${USER}"

# Add other zshrc exports.
cat install_scripts/zshrc-exports >> ~/.zshrc
