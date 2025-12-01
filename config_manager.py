import os
import json

class ConfigManager:
    def __init__(self, master_config_path=None):
        if master_config_path is None:
            from calibration_gui import get_resource_path
            self.master_config_path = get_resource_path("machines_config.json")
        else:
            self.master_config_path = master_config_path
        
        self.machines_config = self.load_master_config()
    
    def load_master_config(self):
        """Load the master configuration file"""
        if not os.path.exists(self.master_config_path):
            raise FileNotFoundError(f"Master config file not found: {self.master_config_path}")
        
        with open(self.master_config_path, 'r') as f:
            return json.load(f)
    
    def load_machines_config(self):
        """Load machines configuration (alias for load_master_config for GUI compatibility)"""
        return self.load_master_config()
    
    def save_machines_config(self, config):
        """Save machines configuration to file - DISABLED for safety"""
        raise PermissionError("Modifying machines_config.json is not allowed. Use local config.json files instead.")
    
    def force_update_local_config(self, machine_id, file_path):
        """Force update of local config with current machine defaults"""
        return self.create_or_update_local_config(machine_id, file_path, force_recreate=True)
    
    def get_available_machines(self):
        """Get list of available machines"""
        return list(self.machines_config["machines"].keys())
    
    def get_machine_name(self, machine_id):
        """Get machine name for a specific machine ID"""
        if machine_id not in self.machines_config["machines"]:
            raise ValueError(f"Machine {machine_id} not found. Available machines: {self.get_available_machines()}")
        
        return self.machines_config["machines"][machine_id]["name"]
    
    def get_machine_config(self, machine_id, file_path):
        """Get configuration for a specific machine"""
        # Always reload machine config to get latest changes
        current_machines_config = self.load_machines_config()
        
        if machine_id not in current_machines_config["machines"]:
            raise ValueError(f"Machine {machine_id} not found. Available machines: {list(current_machines_config['machines'].keys())}")
        
        machine_config = current_machines_config["machines"][machine_id]["default_config"].copy()
        machine_config["file_path"] = file_path
        machine_config["machine_id"] = machine_id
        machine_config["machine_name"] = current_machines_config["machines"][machine_id]["name"]
        
        # Flatten weight_chains structure for GUI editing
        if "weight_chains" in machine_config:
            weight_chains = machine_config.pop("weight_chains")
            
            # Add force weights
            if "force_weights" in weight_chains:
                for key, value in weight_chains["force_weights"].items():
                    machine_config[key] = value
            
            # Add moment weights  
            if "moment_weights" in weight_chains:
                for key, value in weight_chains["moment_weights"].items():
                    machine_config[key] = value
            
            # Add moment distances
            if "moment_distances" in weight_chains:
                for key, value in weight_chains["moment_distances"].items():
                    machine_config[key] = value
            
            # Add gewichtsstufen
            if "gewichtsstufen" in weight_chains:
                machine_config["gewichtsstufen"] = weight_chains["gewichtsstufen"]
        
        return machine_config
    
    def create_or_update_local_config(self, machine_id, file_path, custom_overrides=None, force_recreate=False):
        """Create or update local config file for a specific data file
        
        Args:
            machine_id: Machine identifier (e.g., 'CM1', 'CM2')
            file_path: Path to the data file
            custom_overrides: Dictionary of custom config values to apply
            force_recreate: If True, recreate config from machine defaults ignoring existing file
        """
        folder = os.path.dirname(file_path)
        config_path = os.path.join(folder, "config.json")
        
        # Get current machine default config (this includes any updates from GUI)
        config = self.get_machine_config(machine_id, file_path)
        
        # Apply custom overrides if provided
        if custom_overrides:
            config.update(custom_overrides)
        
        # Load existing config if it exists and force_recreate is False
        if os.path.exists(config_path) and not force_recreate:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
            
            # Check if machine has changed - if so, recreate with new machine defaults
            existing_machine_id = existing_config.get('machine_id')
            if existing_machine_id != machine_id:
                print(f"Machine changed from {existing_machine_id} to {machine_id}, recreating config...")
                # Don't merge - use new machine config entirely
            else:
                # Same machine - preserve existing config values (user may have edited them)
                # Only add missing parameters from machine defaults
                print(f"Preserving existing local config for {machine_id}, adding any missing parameters")
                
                # Add any missing parameters from machine config
                for key, value in config.items():
                    if key not in existing_config:
                        existing_config[key] = value
                        print(f"Added missing parameter: {key} = {value}")
                
                # Use the existing config (preserves user changes)
                config = existing_config
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Config created/updated for {machine_id} at: {config_path}")
        return config

def select_machine():
    """Interactive machine selection"""
    config_manager = ConfigManager()
    machines = config_manager.get_available_machines()
    
    print("Available calibration machines:")
    for i, machine in enumerate(machines, 1):
        machine_name = config_manager.machines_config["machines"][machine]["name"]
        print(f"{i}. {machine} - {machine_name}")
    
    while True:
        try:
            choice = input(f"\nSelect machine (1-{len(machines)}) or type machine ID directly: ").strip()
            
            # Check if it's a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(machines):
                    return machines[idx]
            
            # Check if it's a direct machine ID
            if choice.upper() in machines:
                return choice.upper()
            
            print("Invalid selection. Please try again.")
            
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")