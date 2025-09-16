"""
Unit tests for utility functions.
"""

import unittest
import os
import tempfile
import yaml
from utils.helpers import load_config, save_model, create_experiment_log, ensure_dir_exists


class TestHelpers(unittest.TestCase):
    """Test cases for helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_load_config(self):
        """Test configuration loading."""
        # Create a test config file
        test_config = {
            'data': {'train_path': 'test.csv'},
            'models': {'test_model': {'param': 1}}
        }
        
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Test loading
        loaded_config = load_config(config_path)
        self.assertEqual(loaded_config, test_config)
    
    def test_ensure_dir_exists(self):
        """Test directory creation."""
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        ensure_dir_exists(test_dir)
        self.assertTrue(os.path.exists(test_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
