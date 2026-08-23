"""Model illness, symptoms, treatment, and recovery with a Bayesian network.

The module defines the network structure and conditional probability tables, performs
exact inference with variable elimination, samples approximate results with Gibbs
sampling, and measures and plots how sampling accuracy and runtime change with size.
"""

import time

import matplotlib.pyplot as plt
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import GibbsSampling

# define the network structure
model = DiscreteBayesianNetwork(
    [
        ("flu", "fever"),
        ("covid", "fever"),
        ("flu", "cough"),
        ("covid", "cough"),
        ("covid", "treatment"),
        ("treatment", "recovery"),
        ("fever", "recovery"),
        ("cough", "recovery"),
    ]
)

# define the CPTs
cpd_flu = TabularCPD(variable="flu", variable_card=2, values=[[0.88], [0.12]])
cpd_covid = TabularCPD(variable="covid", variable_card=2, values=[[0.92], [0.08]])
cpt_fever = TabularCPD(
    variable="fever",
    variable_card=2,
    evidence=["flu", "covid"],
    evidence_card=[2, 2],
    values=[[0.99, 0.15, 0.10, 0.02], [0.01, 0.85, 0.90, 0.98]],
)
cpt_cough = TabularCPD(
    variable="cough",
    variable_card=2,
    evidence=["flu", "covid"],
    evidence_card=[2, 2],
    values=[[0.98, 0.40, 0.30, 0.15], [0.02, 0.60, 0.70, 0.85]],
)
cpt_treatment = TabularCPD(
    "treatment",
    variable_card=2,
    evidence=["covid"],
    evidence_card=[2],
    values=[[0.95, 0.05], [0.05, 0.95]],
)
cpd_recovery = TabularCPD(
    variable="recovery",
    variable_card=2,
    evidence=["cough", "fever", "treatment"],
    evidence_card=[2, 2, 2],
    values=[
        [0.99, 0.99, 0.50, 0.90, 0.85, 0.85, 0.80, 0.30],
        [0.01, 0.01, 0.50, 0.10, 0.15, 0.15, 0.20, 0.70],
    ],
)

model.add_cpds(cpd_flu, cpd_covid, cpt_fever, cpt_cough, cpt_treatment, cpd_recovery)
assert model.check_model()
inference = VariableElimination(model)

print("Exact Inference")

# perform the inference queries with the appropriate evidence
print("P(covid) | fever=True, cough=True")
print(inference.query(variables=["covid"], evidence={"fever": 1, "cough": 1}))
print("P(flu) | fever=True, cough=False")
print(inference.query(variables=["flu"], evidence={"fever": 1, "cough": 0}))
print("P(treatment) | cough=True")
print(inference.query(variables=["treatment"], evidence={"cough": 1}))
print("P(recovery) | fever=True, treatment=True")
print(inference.query(variables=["recovery"], evidence={"fever": 1, "treatment": 1}))

inference = GibbsSampling(model)
print("Gibbs Sampling")

# generate the gibbs samples
samples = inference.sample(size=10000, seed=42)

# calculate the conditional probabilities by filtering the dataframe and normalizing
print("P(covid) | fever=True, cough=True")
covid_filter = samples[(samples["fever"] == 1) & (samples["cough"] == 1)]
covid_counts = covid_filter["covid"].value_counts(normalize=True).sort_index()
print(covid_counts)
print("P(flu) | fever=True, cough=False")
flu_filter = samples[(samples["fever"] == 1) & (samples["cough"] == 0)]
flu_counts = flu_filter["flu"].value_counts(normalize=True).sort_index()
print(flu_counts)
print("P(treatment) | cough=True")
treatment_filter = samples[samples["cough"] == 1]
treatment_counts = (
    treatment_filter["treatment"].value_counts(normalize=True).sort_index()
)
print(treatment_counts)
print("P(recovery) | fever=True, treatment=True")
recovery_filter = samples[(samples["fever"] == 1) & (samples["treatment"] == 1)]
recovery_counts = recovery_filter["recovery"].value_counts(normalize=True).sort_index()
print(recovery_counts)


def measure_runtime(
    model: DiscreteBayesianNetwork, num_samples: int
) -> tuple[float, float]:
    """
    Measure the run time of both approaches, and return a tuple of elimination time, gibbs time
    """

    start_time = time.time()
    inference = VariableElimination(model)

    # perform the inference queries with the appropriate evidence
    inference.query(variables=["covid"], evidence={"fever": 1, "cough": 1})
    inference.query(variables=["flu"], evidence={"fever": 1, "cough": 0})
    inference.query(variables=["treatment"], evidence={"cough": 1})
    inference.query(variables=["recovery"], evidence={"fever": 1, "treatment": 1})
    elim_time = time.time() - start_time

    start_time = time.time()
    inference = GibbsSampling(model)

    # generate the gibbs samples
    samples = inference.sample(size=num_samples, seed=42)

    # calculate the conditional probabilities by filtering the dataframe and normalizing
    covid_filter = samples[(samples["fever"] == 1) & (samples["cough"] == 1)]
    covid_counts = covid_filter["covid"].value_counts(normalize=True).sort_index()
    flu_filter = samples[(samples["fever"] == 1) & (samples["cough"] == 0)]
    flu_counts = flu_filter["flu"].value_counts(normalize=True).sort_index()
    treatment_filter = samples[samples["cough"] == 1]
    treatment_counts = (
        treatment_filter["treatment"].value_counts(normalize=True).sort_index()
    )
    recovery_filter = samples[(samples["fever"] == 1) & (samples["treatment"] == 1)]
    recovery_counts = (
        recovery_filter["recovery"].value_counts(normalize=True).sort_index()
    )
    gibbs_time = time.time() - start_time

    return elim_time, gibbs_time


# define the network structure
model_XL = DiscreteBayesianNetwork(
    [
        ("flu", "fever"),
        ("covid", "fever"),
        ("flu", "cough"),
        ("covid", "cough"),
        ("flu", "dizzy"),
        ("covid", "dizzy"),
        ("covid", "treatment"),
        ("treatment", "recovery"),
        ("fever", "recovery"),
        ("cough", "recovery"),
        ("dizzy", "recovery"),
    ]
)

# define the CPTs
cpd_flu = TabularCPD(variable="flu", variable_card=2, values=[[0.88], [0.12]])
cpd_covid = TabularCPD(variable="covid", variable_card=2, values=[[0.92], [0.08]])
cpt_fever = TabularCPD(
    variable="fever",
    variable_card=2,
    evidence=["flu", "covid"],
    evidence_card=[2, 2],
    values=[[0.99, 0.15, 0.10, 0.02], [0.01, 0.85, 0.90, 0.98]],
)
cpt_cough = TabularCPD(
    variable="cough",
    variable_card=2,
    evidence=["flu", "covid"],
    evidence_card=[2, 2],
    values=[[0.98, 0.40, 0.30, 0.15], [0.02, 0.60, 0.70, 0.85]],
)
cpt_dizzy = TabularCPD(
    variable="dizzy",
    variable_card=2,
    evidence=["flu", "covid"],
    evidence_card=[2, 2],
    values=[[0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4]],
)
cpt_treatment = TabularCPD(
    "treatment",
    variable_card=2,
    evidence=["covid"],
    evidence_card=[2],
    values=[[0.95, 0.05], [0.05, 0.95]],
)
cpd_recovery = TabularCPD(
    variable="recovery",
    variable_card=2,
    evidence=["dizzy", "cough", "fever", "treatment"],
    evidence_card=[2, 2, 2, 2],
    values=[
        [
            0.99,
            0.99,
            0.50,
            0.90,
            0.85,
            0.85,
            0.80,
            0.30,
            0.99,
            0.92,
            0.59,
            0.70,
            0.20,
            0.17,
            0.45,
            0.66,
        ],
        [
            0.01,
            0.01,
            0.50,
            0.10,
            0.15,
            0.15,
            0.20,
            0.70,
            0.01,
            0.08,
            0.41,
            0.30,
            0.80,
            0.83,
            0.55,
            0.34,
        ],
    ],
)

model_XL.add_cpds(
    cpd_flu, cpd_covid, cpt_dizzy, cpt_fever, cpt_cough, cpt_treatment, cpd_recovery
)
assert model_XL.check_model()

# original model
baseline = measure_runtime(model=model, num_samples=10000)

# augmented model
augmented = measure_runtime(model=model_XL, num_samples=10000)

print(baseline)
print(augmented)

data = [
    (0.0062656402587890625, 20.362045526504517),
    (0.009830951690673828, 11.184540033340454),
]

first_elements = [x for x, _ in data]
second_elements = [y for _, y in data]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# First plot
ax1.plot(first_elements, marker="o", linestyle="-", color="blue")
ax1.set_title("Variable Elimination")
ax2.set_xlabel("Baseline and Augmented Model")
ax2.set_ylabel("Runtime")
ax1.grid(True)

# Second plot
ax2.plot(second_elements, marker="o", linestyle="-", color="green")
ax2.set_title("Gibbs Sampling")
ax2.set_xlabel("Baseline and Augmented Model")
ax2.set_ylabel("Runtime")
ax2.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
