from src.pipline.training_pipeline import TrainingPipeline

TrainingPipeline().run_pipeline()

'''
from src.pipline.prediction_pipeline import PredictionPipeline

obj = PredictionPipeline()

data = {
    'age' : 38,
    'gender' : 'female',
    'marital_status' : 'Married',
    'education' : 'Professional',
    'monthly_salary' : 82600,
    'employment_type' : 'Private',
    'years_of_employment' : 0.9,
    'company_type' : 'Mid-size',
    'house_type' : 'Rented',
    'monthly_rent' : 20000,
    'family_size' : 3,
    'dependents' : 2,
    'school_fees' : 0,
    'college_fees' : 0,
    'travel_expenses' : 7200,
    'groceries_utilities' : 19500,
    'other_monthly_expenses' : 13200,
    'existing_loans' : 'Yes', 
    'current_emi_amount' : 23700,  
    'credit_score' : 660,
    'bank_balance' : 303200,
    'emergency_fund' : 70200,
    'emi_scenario' : 'Personal Loan EMI',
    'requested_amount' : 850000,
    'requested_tenure' : 15
}

print(obj.predict(data))
'''