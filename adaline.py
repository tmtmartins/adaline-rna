import random

def generate_synaptic_weights():
    return [round(random.uniform(0, 1), 5) for _ in range(5)]

def calculate_sample(synaptic_weights, values):
    u = sum(weight * value for weight, value in zip(synaptic_weights, values))
    return u

def update_weights(weights, learning_rate, u, row):
    return [(old_weight + learning_rate * (row['d'] - u) * row[f'x{index}']) for index, old_weight in enumerate(weights)]


def eqm_difference_is_less_than_precision(eqm, old_eqm, precision):
    result = (abs(eqm - old_eqm))
    if (result <= precision):
        return True
    return False

def calculate_eqm(samples):
    eqm = 0
    for sample in samples:
        eqm += ((sample['d'] - sample['u']) * (sample['d'] - sample['u']))
    return eqm/len(samples)

def train_model(data, learning_rate, interactions_amount_limit, precision):
    interactions_amount = 0
    has_converged = False
    synaptic_weights = generate_synaptic_weights()
    print(f"Pesos iniciais: {synaptic_weights}")
    old_eqm = 0
    eqm = 0
    while not (has_converged or (interactions_amount >= interactions_amount_limit)):
        eqm_json = []
        for index, row in data.iterrows():
            sample = calculate_sample(synaptic_weights, row)
            synaptic_weights = update_weights(synaptic_weights, learning_rate, sample, row)
            eqm_json.append({'d': row['d'], 'u': sample})
        interactions_amount += 1
        old_eqm = eqm
        eqm = calculate_eqm(eqm_json)
        has_converged = eqm_difference_is_less_than_precision(eqm, old_eqm, precision)
    
    print(f"Epocas: {interactions_amount}\nPesos Finais: {synaptic_weights}\nConvergiu: {has_converged}")
    return synaptic_weights

def run_tests(synaptic_weights, df):
    predicted_values = []
    for index, row in df.iterrows():
        sample = calculate_sample(synaptic_weights, row)
        predicted_values.append("B" if sample > 0 else "A")
    df['d_predicted'] = predicted_values
    return df

def run(training_df, test_df, learning_rate, interactions_amount_limit, precision):
    conveged_synaptic_weights = train_model(training_df, learning_rate, interactions_amount_limit, precision)
    print(run_tests(conveged_synaptic_weights, test_df))
