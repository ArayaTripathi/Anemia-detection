// Anemia Detection Medical Dashboard Script

document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const imageInput = document.getElementById('imageInput');
  const previewContainer = document.getElementById('previewContainer');
  const previewImg = document.getElementById('previewImg');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const removeImgBtn = document.getElementById('removeImgBtn');
  const form = document.getElementById('anemiaForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const samplePills = document.querySelectorAll('.sample-pill');
  const newAnalysisBtn = document.getElementById('newAnalysisBtn');
  const resultsTabBtn = document.getElementById('results-tab');
  const inputTabBtn = document.getElementById('input-tab');

  // Drag and drop events
  if (dropzone && imageInput) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        imageInput.files = files;
        handleFileSelect(files[0]);
      }
    });

    imageInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
      alert('Please upload a valid image file (PNG, JPG, JPEG, TIFF)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewContainer.classList.remove('d-none');
      dropzonePrompt.classList.add('d-none');
    };
    reader.readAsDataURL(file);
  }

  // Remove preview
  if (removeImgBtn) {
    removeImgBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      imageInput.value = '';
      previewImg.src = '';
      previewContainer.classList.add('d-none');
      dropzonePrompt.classList.remove('d-none');
    });
  }

  // Quick sample images loader
  samplePills.forEach(pill => {
    pill.addEventListener('click', async () => {
      const sampleFile = pill.getAttribute('data-sample');
      const sampleName = pill.getAttribute('data-name');
      const sampleHb = pill.getAttribute('data-hb');
      const sampleMcv = pill.getAttribute('data-mcv');
      const sampleRbc = pill.getAttribute('data-rbc');
      const sampleMch = pill.getAttribute('data-mch');

      // Autofill CBC fields if available
      if (sampleHb) document.getElementById('hemoglobin').value = sampleHb;
      if (sampleMcv) document.getElementById('mcv').value = sampleMcv;
      if (sampleRbc) document.getElementById('rbc_count').value = sampleRbc;
      if (sampleMch) document.getElementById('mch').value = sampleMch;

      try {
        const response = await fetch(`/static/samples/${sampleFile}`);
        const blob = await response.blob();
        const file = new File([blob], sampleFile, { type: blob.type || 'image/jpeg' });
        
        // Use DataTransfer to set file input
        const dt = new DataTransfer();
        dt.items.add(file);
        imageInput.files = dt.files;
        handleFileSelect(file);
      } catch (err) {
        console.error('Error loading sample image:', err);
      }
    });
  });

  // Start new analysis button
  if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener('click', () => {
      if (inputTabBtn) {
        const bsTab = new bootstrap.Tab(inputTabBtn);
        bsTab.show();
      }
    });
  }

  // Form submission with loading spinner
  if (form) {
    form.addEventListener('submit', async (e) => {
      if (!imageInput.files || imageInput.files.length === 0) {
        e.preventDefault();
        alert('Please select or upload a blood smear image.');
        return;
      }

      // Show spinner
      if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
      }
      // Form submits to /predict
    });
  }
});

