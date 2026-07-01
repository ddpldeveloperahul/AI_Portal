import os
import shutil
import tempfile
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import resolve, reverse

from .models import PredictionTask
from .views import run_prediction, task_status, upload_image


class RoadDetectionUrlTests(SimpleTestCase):
    def test_upload_url_matches_frontend_endpoint(self):
        match = resolve('/road/road-detection/upload/')

        self.assertEqual(match.func, upload_image)
        self.assertEqual(reverse('upload_image'), '/road/road-detection/upload/')

    def test_status_url_matches_frontend_endpoint(self):
        match = resolve('/road/road-detection/status/1/')

        self.assertEqual(match.func, task_status)
        self.assertEqual(reverse('task_status', args=[1]), '/road/road-detection/status/1/')


class RoadPredictionTests(TestCase):
    def setUp(self):
        self.media_root = tempfile.mkdtemp()
        self.settings_override = override_settings(MEDIA_ROOT=self.media_root)
        self.settings_override.enable()

    def tearDown(self):
        self.settings_override.disable()
        shutil.rmtree(self.media_root)

    def test_run_prediction_creates_output_directory(self):
        task = PredictionTask.objects.create(
            input_file=SimpleUploadedFile('road.tif', b'test data'),
            status='PROCESSING',
        )

        with patch('road_app.views.process_image') as mock_process_image:
            run_prediction(task.id)

        output_dir = os.path.join(self.media_root, 'outputs')
        expected_output = os.path.join(output_dir, f'prediction_{task.id}.tif')
        expected_png = os.path.join(output_dir, f'prediction_{task.id}.png')
        expected_shapefile = os.path.join(output_dir, f'prediction_{task.id}_shapefile.zip')

        self.assertTrue(os.path.isdir(output_dir))
        mock_process_image.assert_called_once()
        self.assertEqual(mock_process_image.call_args.args[1], expected_output)
        self.assertEqual(mock_process_image.call_args.kwargs['png_output_path'], expected_png)
        self.assertEqual(mock_process_image.call_args.kwargs['shapefile_zip_path'], expected_shapefile)

        task.refresh_from_db()
        self.assertEqual(task.status, 'COMPLETED')
        self.assertEqual(task.output_file.name, f'outputs/prediction_{task.id}.tif')
