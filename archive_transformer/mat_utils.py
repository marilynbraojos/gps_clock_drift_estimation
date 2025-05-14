import scipy.io

class MatUtils: 
    def __init__(self, file_path):
        self.file_path = file_path 
        self.mat_data = self.load_mat_file()

    def load_mat_file(self): 
        """Load the .mat file and return its contents."""
        try: 
            return scipy.io.loadmat(self.file_path)
        except Exception as e: 
            print(f"Error loading .mat file: {e}.")
            return None

    def print_mat_file_content(self):
        """Print the contents of the loaded .mat file"""
        if not self.mat_data: 
            print("No data to show. Check filepath.")
            return 

        print(f"Contents of {self.file_path}:")
        for key, value in self.mat_data.items():
            if not key.startswith('__'):  # skip the metadata keys
                print(f"Variable name: {key}, Shape: {value.shape}")