document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    lucide.createIcons();
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const predictionLabel = document.getElementById('prediction-label');
    const riskCategoryLabel = document.getElementById('risk-category');
    const riskScoreLabel = document.getElementById('risk-score');
    const navLinks = document.querySelectorAll('.nav-links a');

    let riskGaugeChart = null;
    let importanceChartLoaded = false;

    // Landing Page Transition
    const landingPage = document.getElementById('landing-page');
    const dashboardContent = document.getElementById('dashboard-content');
    const startBtn = document.getElementById('start-discovery');

    startBtn.addEventListener('click', () => {
        transitionToDashboard();
    });

    // Add skip to gradio listener if exists
    const skipBtn = document.getElementById('skip-to-gradio');
    if (skipBtn) {
        skipBtn.addEventListener('click', (e) => {
            e.preventDefault();
            transitionToDashboard('gradio-view');
        });
    }

    function transitionToDashboard(targetSection = 'hub') {
        landingPage.style.opacity = '0';
        landingPage.style.pointerEvents = 'none';
        setTimeout(() => {
            landingPage.style.visibility = 'hidden';
            dashboardContent.classList.add('active');
            showSection(targetSection);
            // If landing on insights directly, load chart now
            if (targetSection === 'insights') {
                setTimeout(loadImportanceChart, 50);
            }
        }, 800);
    }

    // Feature Hub Navigation
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('click', () => {
            const target = card.getAttribute('data-target');
            showSection(target);
        });
    });

    // Back Buttons
    document.querySelectorAll('.btn-back').forEach(btn => {
        btn.addEventListener('click', () => {
            showSection('hub');
        });
    });

    function showSection(sectionId) {
        document.querySelectorAll('section').forEach(sec => {
            sec.classList.add('hidden');
            sec.classList.remove('active');
        });
        const target = document.getElementById(sectionId);
        if (target) {
            target.classList.remove('hidden');
            setTimeout(() => target.classList.add('active'), 10);
        }

        // Lazy-load feature importance chart when insights section becomes visible
        if (sectionId === 'insights' && !importanceChartLoaded) {
            setTimeout(loadImportanceChart, 100);
        }

        // Update Nav Links
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${sectionId}`) {
                link.classList.add('active');
            }
        });

        // Special handling for hub/dashboard default
        if (sectionId === 'hub') {
            document.querySelectorAll('.nav-links a').forEach(l => {
                if (l.getAttribute('href') === '#hub') l.classList.add('active');
            });
        }
    }

    // Top Nav Link Click
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('href').substring(1);
            showSection(target || 'hub');
        });
    });

    // Gradio Toggle Listener
    const gradioToggle = document.querySelector('.gradio-toggle');
    if (gradioToggle) {
        gradioToggle.addEventListener('click', (e) => {
            e.preventDefault();
            showSection('gradio-view');
        });
    }

    // Load Feature Importance Chart on Start (Moved to button click trigger)

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // Convert numeric values
            if (key === 'tenure' || key === 'SeniorCitizen') {
                data[key] = parseInt(value);
            } else if (key === 'MonthlyCharges' || key === 'TotalCharges') {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        });

        const submitBtn = form.querySelector('button');
        submitBtn.innerText = 'Analyzing...';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            displayResult(result);
        } catch (error) {
            console.error('Error:', error);
            alert('Service unavailable. Please ensure the backend is running.');
        } finally {
            submitBtn.innerText = 'Calculate Success Probability';
            submitBtn.disabled = false;
        }
    });

    function displayResult(result) {
        resultContainer.classList.remove('hidden');
        predictionLabel.innerText = result.prediction === 'Churn' ? 'High Churn Risk' : 'Healthy Customer';
        riskCategoryLabel.innerText = `Risk Priority: ${result.risk_category}`;
        riskScoreLabel.innerText = `${Math.round(result.probability * 100)}%`;

        // Update Gauge Color based on risk
        const color = result.probability > 0.6 ? '#FF4D4D' : (result.probability > 0.3 ? '#F59E0B' : '#10B981');

        updateGauge(result.probability * 100, color);

        // Scroll to result
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function updateGauge(value, color) {
        const ctx = document.getElementById('riskGauge').getContext('2d');

        if (riskGaugeChart) {
            riskGaugeChart.destroy();
        }

        riskGaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [value, 100 - value],
                    backgroundColor: [color, 'rgba(255,255,255,0.05)'],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270,
                    cutout: '85%',
                    borderRadius: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            }
        });
    }

    async function loadImportanceChart() {
        if (importanceChartLoaded) return;
        try {
            const response = await fetch('/feature-importance');
            const data = await response.json();
            const importances = data.feature_importance;

            const canvas = document.getElementById('importanceChart');
            const ctx = canvas.getContext('2d');

            const gradient = ctx.createLinearGradient(0, 0, 400, 0);
            gradient.addColorStop(0, '#A78BFA'); // Lavender
            gradient.addColorStop(1, '#34D399'); // Mint Green

            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: importances.map(i => i.feature),
                    datasets: [{
                        label: 'Impact Score',
                        data: importances.map(i => i.importance),
                        backgroundColor: gradient,
                        borderRadius: 6,
                        barThickness: 20
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: false },
                        y: {
                            ticks: { color: '#9CA3AF', font: { family: 'Outfit', size: 13 } },
                            grid: { display: false }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
            importanceChartLoaded = true;
        } catch (error) {
            console.error('Error loading insights:', error);
        }
    }

    // Batch Upload Mock (Real implementation would use /predict-batch)
    const dropZone = document.getElementById('drop-zone');
    const csvInput = document.getElementById('csv-upload');

    dropZone.addEventListener('click', () => csvInput.click());

    csvInput.addEventListener('change', async () => {
        if (csvInput.files.length > 0) {
            const file = csvInput.files[0];
            const formData = new FormData();
            formData.append('file', file);

            const uploadText = dropZone.querySelector('p');
            uploadText.innerText = 'Processing Batch...';

            try {
                const response = await fetch('/predict-batch', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                displayBatchResults(result.batch_results);
            } catch (error) {
                alert('Batch processing failed.');
            } finally {
                uploadText.innerHTML = 'Drop CSV files here or <span>click to upload</span>';
            }
        }
    });

    function displayBatchResults(results) {
        const counts = { Low: 0, Medium: 0, High: 0 };
        results.forEach(r => counts[r.risk_category]++);

        const batchContainer = document.getElementById('batch-results');
        batchContainer.classList.remove('hidden');

        const ctx = document.getElementById('batchChart').getContext('2d');
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    data: [counts.Low, counts.Medium, counts.High],
                    backgroundColor: ['#10B981', '#F59E0B', '#FF4D4D'],
                    borderWidth: 0,
                    hoverOffset: 15
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#9CA3AF', font: { family: 'Outfit' }, padding: 20 } }
                }
            }
        });
        batchContainer.scrollIntoView({ behavior: 'smooth' });
    }
});
