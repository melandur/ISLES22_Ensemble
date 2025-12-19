"""
Unit tests for main.py argument parsing and validation.
"""
import os
import pytest
import sys
from unittest.mock import patch, Mock
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMainArgumentParsing:
    """Tests for argument parsing in main.py."""
    
    def test_parse_required_arguments(self):
        """Test parsing required arguments."""
        from main import main
        
        # Mock sys.argv
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz',
            '--adc_file_name', 'test_adc.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            with patch('os.path.exists', return_value=True):
                with patch('os.makedirs'):
                    with patch('main.IslesEnsemble') as mock_ensemble:
                        mock_instance = Mock()
                        mock_ensemble.return_value = mock_instance
                        
                        with patch('subprocess.run'), \
                             patch('src.isles22_ensemble.convert_to_nii') as mock_convert, \
                             patch('src.isles22_ensemble.get_img_shape', return_value=3), \
                             patch('src.isles22_ensemble.nib.load') as mock_nib:
                            # Create a mock NIfTI image with proper shape
                            mock_img = Mock()
                            mock_img.shape = (64, 64, 64)
                            mock_nib.return_value = mock_img
                            mock_convert.side_effect = [
                                ('/app/data/test_dwi.nii.gz', True),
                                ('/app/data/test_adc.nii.gz', True),
                            ]
                            main()
                            
                            # Verify ensemble was called
                            mock_instance.predict_ensemble.assert_called_once()
    
    def test_parse_all_arguments(self):
        """Test parsing all available arguments."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz',
            '--adc_file_name', 'test_adc.nii.gz',
            '--flair_file_name', 'test_flair.nii.gz',
            '--fast',
            '--save_team_outputs',
            '--skull_strip',
            '--results_mni'
        ]
        
        with patch('sys.argv', test_args):
            with patch('os.path.exists', return_value=True):
                with patch('os.makedirs'):
                    with patch('main.IslesEnsemble') as mock_ensemble:
                        mock_instance = Mock()
                        mock_ensemble.return_value = mock_instance
                        
                        with patch('subprocess.run'), \
                             patch('src.isles22_ensemble.convert_to_nii') as mock_convert, \
                             patch('src.isles22_ensemble.get_img_shape', return_value=3), \
                             patch('src.isles22_ensemble.nib.load') as mock_nib:
                            mock_img = Mock()
                            mock_img.shape = (64, 64, 64)
                            mock_nib.return_value = mock_img
                            mock_convert.side_effect = [
                                ('/app/data/test_dwi.nii.gz', True),
                                ('/app/data/test_adc.nii.gz', True),
                            ]
                            main()
                            
                            # Verify ensemble was called with correct arguments
                            call_kwargs = mock_instance.predict_ensemble.call_args[1]
                            assert call_kwargs['fast'] is True
                            assert call_kwargs['save_team_outputs'] is True
                            assert call_kwargs['skull_strip'] is True
                            assert call_kwargs['results_mni'] is True
    
    def test_parse_optional_flair(self):
        """Test parsing with optional FLAIR argument."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz',
            '--adc_file_name', 'test_adc.nii.gz',
            '--flair_file_name', 'test_flair.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            with patch('os.path.exists', return_value=True):
                with patch('os.makedirs'):
                    with patch('main.IslesEnsemble') as mock_ensemble:
                        mock_instance = Mock()
                        mock_ensemble.return_value = mock_instance
                        
                        with patch('subprocess.run'), \
                             patch('src.isles22_ensemble.convert_to_nii') as mock_convert, \
                             patch('src.isles22_ensemble.get_img_shape', return_value=3), \
                             patch('src.isles22_ensemble.nib.load') as mock_nib:
                            mock_img = Mock()
                            mock_img.shape = (64, 64, 64)
                            mock_nib.return_value = mock_img
                            mock_convert.side_effect = [
                                ('/app/data/test_dwi.nii.gz', True),
                                ('/app/data/test_adc.nii.gz', True),
                                ('/app/data/test_flair.nii.gz', True),
                            ]
                            main()
                            
                            call_kwargs = mock_instance.predict_ensemble.call_args[1]
                            assert call_kwargs['input_flair_path'] is not None


class TestMainFileValidation:
    """Tests for file validation in main.py."""
    
    def test_validate_dwi_file_exists(self):
        """Test validation when DWI file exists."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz',
            '--adc_file_name', 'test_adc.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            with patch('os.path.exists', return_value=True):
                with patch('os.makedirs'):
                    with patch('main.IslesEnsemble') as mock_ensemble:
                        mock_instance = Mock()
                        mock_ensemble.return_value = mock_instance
                        
                        with patch('subprocess.run'), \
                             patch('src.isles22_ensemble.convert_to_nii') as mock_convert, \
                             patch('src.isles22_ensemble.get_img_shape', return_value=3), \
                             patch('src.isles22_ensemble.nib.load') as mock_nib:
                            mock_img = Mock()
                            mock_img.shape = (64, 64, 64)
                            mock_nib.return_value = mock_img
                            mock_convert.side_effect = [
                                ('/app/data/test_dwi.nii.gz', True),
                                ('/app/data/test_adc.nii.gz', True),
                            ]
                            # Should not raise
                            main()
    
    def test_validate_dwi_file_not_exists(self):
        """Test error when DWI file does not exist."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'nonexistent_dwi.nii.gz',
            '--adc_file_name', 'test_adc.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            def mock_exists(path):
                if 'nonexistent' in path:
                    return False
                return True
            
            with patch('os.path.exists', side_effect=mock_exists):
                with pytest.raises(FileNotFoundError):
                    main()
    
    def test_validate_adc_file_not_exists(self):
        """Test error when ADC file does not exist."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz',
            '--adc_file_name', 'nonexistent_adc.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            def mock_exists(path):
                if 'nonexistent' in path:
                    return False
                return True
            
            with patch('os.path.exists', side_effect=mock_exists):
                with pytest.raises(FileNotFoundError):
                    main()
    
    def test_validate_flair_file_not_exists(self):
        """Test error when FLAIR file does not exist."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz',
            '--adc_file_name', 'test_adc.nii.gz',
            '--flair_file_name', 'nonexistent_flair.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            def mock_exists(path):
                if 'nonexistent' in path:
                    return False
                return True
            
            with patch('os.path.exists', side_effect=mock_exists):
                with pytest.raises(FileNotFoundError):
                    main()


class TestMainErrorHandling:
    """Tests for error handling in main.py."""
    
    def test_missing_dwi_argument(self):
        """Test error when DWI argument is missing."""
        from main import main
        
        test_args = [
            'main.py',
            '--adc_file_name', 'test_adc.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):  # argparse raises SystemExit
                main()
    
    def test_missing_adc_argument(self):
        """Test error when ADC argument is missing."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):  # argparse raises SystemExit
                main()
    
    def test_output_directory_creation(self):
        """Test that output directory is created."""
        from main import main
        
        test_args = [
            'main.py',
            '--dwi_file_name', 'test_dwi.nii.gz',
            '--adc_file_name', 'test_adc.nii.gz'
        ]
        
        with patch('sys.argv', test_args):
            with patch('os.path.exists', return_value=True):
                with patch('os.makedirs') as mock_makedirs:
                    with patch('main.IslesEnsemble') as mock_ensemble:
                        mock_instance = Mock()
                        mock_ensemble.return_value = mock_instance
                        
                        with patch('subprocess.run'), \
                             patch('src.isles22_ensemble.convert_to_nii') as mock_convert, \
                             patch('src.isles22_ensemble.get_img_shape', return_value=3), \
                             patch('src.isles22_ensemble.nib.load') as mock_nib:
                            mock_img = Mock()
                            mock_img.shape = (64, 64, 64)
                            mock_nib.return_value = mock_img
                            mock_convert.side_effect = [
                                ('/app/data/test_dwi.nii.gz', True),
                                ('/app/data/test_adc.nii.gz', True),
                            ]
                            main()
                            
                            # Verify output directory was created
                            mock_makedirs.assert_called()

