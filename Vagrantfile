Vagrant.configure("2") do |config|
  config.vm.box = "bento/ubuntu-16.04"
  # Set up machine
  config.vm.provision :shell, path: "bootstrap.sh"

  config.vm.provider "virtualbox" do |vb|
    # Display the VirtualBox GUI when booting the machine
    vb.gui = true

    # Customize the amount of memory on the VM:
    vb.memory = "2048"
  end
end
