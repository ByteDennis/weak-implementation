from skweak.utils import list_available_models, make_weak_classification
from skweak.experiments import EndModelExperiment 

print(list_available_models())

# Generate synthetic data
X, L, y = make_weak_classification(n_samples=1000, n_sources=10)

# Run experiment
experiment = EndModelExperiment(label_model='mv', end_model='mlp')
results = experiment.run_experiment(X, L, y)
print(f"Test Accuracy: {results['test_accuracy']:.3f}")