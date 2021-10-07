
less_equal = r"""$$\le$$"""

def landing_header():
    text = \
    """
    # Sample Calculators 🧮
    **Calculate a model's performance based on sampled data.**

    Use these calculators to:  
    * 🔬 Interpret sample results.  
    * 💰 Plan an sample budget - determine an sample size, by exploring trade-offs.

    When interpreting sample results 🔬, these calculators may assess the following: 
    * 📐 Determining Accuracy: *"How well does the model perform, and range of confidence?"* 
    * ⛑ Ensuring Value Clearance: *"Does the model perform better than value X?"* 
    * ⚖️ Model Comparison: *"Is the model performing better than a previous?"*
    """
    # We find that understanding how to draw conclusions from a sample 🔬, assists budgeting for the required
    #     sample size 💰.


    return text

def landing_instructions():
    text = \
    """
    ## Let's Go!
    To start, simply select on the left hand panel ⬅️  what you are interesting in calculating:

    * Sample **Interpretation** 🔬 or **Budgeting** 💰?

    After which you will be able to a select model question to answer:
    * **Determine Accuracy** 📐 or **Value Clearance** ⛑?

    🚧🚧🚧🚧🚧\n
    Apologies, but the **Model Comparison** option ⚖️ is not available for **Budgeting** 💰 (but is for **Interpretation** 🔬!).\n
    🚧🚧🚧🚧🚧
    """

    return text




def interpret_pass_header(success_rate_boundary, metric_name="success"):
    text = \
    f"""
    # Sample Value Clearance Interpreter ⛑🔬️ 
    **Got data? Use this calculator to draw conclusions about clearance of {success_rate_boundary * 100.:0.1f}% {metric_name}.**

    This **Interpreter** calculator addresses the question:   
    “Given an **Sample Size** with an **Sample {metric_name.title()} Rate**, how likely is the the generating model to be >{success_rate_boundary * 100.:0.1f}% {metric_name}?”


    ### Instructions  
    ⬅️ Please provide on the left hand panel the **Sample {metric_name.title()} Rate**, **Sample Size** and **Risk Factor**.
    """

    return text


def interpret_accuracy_header(ci_fraction, metric_name="success"):
    text = \
    f"""
    # Sample Accuracy Interpreter 📐🔬️ 
    **Got data? Use this calculator to assess the {ci_fraction * 100.:0.1f}% Credible Interval of {metric_name}.**

    This **Interpreter** calculator addresses the question:   
    “Given an **Sample Size** with an **Sample {metric_name.title()} Rate**, what is the model {metric_name} rate within a {ci_fraction * 100.:0.1f}% credible interval?”


    ### Instructions  
    ⬅️ Please provide on the left hand panel the **Sample {metric_name.title()} Rate** and **Sample Size**.
    """

    return text


def interpret_comparison_header(metric_name="success"):
    text = \
    f"""
    # Model Comparison Interpreter ⚖️🔬️ 
    **Got data? Use this calculator to compare the performance of two models.**

    This **Interpreter** calculator addresses the question:   
    “Given sampled data of two models - how do they compare?”

    ### Instructions  
    ⬅️ Please provide on the left hand panel the **Sample {metric_name.title()} Rates** and **Sample Sizes** of Models **A** and **B**.
    """

    return text

def plan_accuracy_header(metric_name="success"):
    text = \
    f"""
    # Sample Accuracy Planner 📐💰

    **Plan an sample budget based on the desired accuracy.**   

    This calculator addresses the question:  
    “What is the minimum **Sample Size** necessary to determine a **{metric_name.title()} Rate** value?“ 

    Two values are required to determine the **Sample Size**:
    * **Goal Accuracy** - The more accurate the result, the larger the sample size required. 
    * **Baseline {metric_name.title()} Rate** - The expected metric value. The closer this expected value is to 50% the larger the sample size required.
    """

    return text


def plan_clearance_header(success_rate_boundary, metric_name="success"):
    text = \
    f"""
    # Sample Clearance Planner ⛑️💰

    **Plan an sample budget to ensure enough data to determine that the model {metric_name} rate is better than {success_rate_boundary * 100:0.1f}%.**   

    This **Planning** calculator addresses the question:   
    “What **Sample Size** should be budgeted for an expected **Sample {metric_name.title()} Rate**, to ensure that the generating model to be >{success_rate_boundary * 100.:0.1f}% {metric_name}?”

    Two values are required to determine the **Sample Size**:
    * **Risk Factor** - The maximum  **False Positive Rate** deemed acceptable. 
    * **Sample {metric_name.title()} Rate** - The min sample {metric_name} rate to ensure **Sample FPR** < **Risk Factor**. 

    ### Instructions  
    ⬅️ On the left hand panel provide the **Risk Factor** expected and the **Sample {metric_name.title()} Rate** to find out the minimum **Sample Size**.
    """

    return text

def risk_factor_explanation(success_rate_boundary, mfpr_rate, metric_name="success"):
    text = f"""### Risk Factor Explanation
*A model passes >{success_rate_boundary * 100.:0.1f}% {metric_name} if the **Sample FPR**{less_equal}{mfpr_rate * 100:0.2f}%.*   

*This guarantees, e.g, that for every 1,000 similar sample results, we consider a maximum of
{mfpr_rate * 1000.:0.0f} **incorrect** "model pass decisions" to be acceptable. (I.e,  models that are actually {less_equal}{success_rate_boundary * 100.:0.1f}% {metric_name})*
"""

    return text

def observed_fpr_to_reason_fpr(observed_fpr, mfpr_rate):
    if observed_fpr < 0.001:
        reason_str = f"<{mfpr_rate * 100.:0.2f}%"
    else:
        str_observed_fpr = f"{observed_fpr * 100:0.1f}%"

        if observed_fpr == mfpr_rate:
            f"{str_observed_fpr}={mfpr_rate * 100.:0.2f}%"
        else:
            reason_str = f"{str_observed_fpr}<{mfpr_rate * 100.:0.2f}%"

    return reason_str

def risk_success(success_rate_boundary, observation_result, mfpr_rate, metric_name="success"):
    reason_str = observed_fpr_to_reason_fpr(observation_result['false_rate'], mfpr_rate)

    text = f""" The model that generated this 
                                    sample result **may be considered >{success_rate_boundary*100.:0.1f}% {metric_name}**! 🎉🎈🎊 
        
**Reason**  
**Sample FPR** is **smaller** than the **Risk Factor** 
                                    ({reason_str}). 
                                    """

    return text

def risk_fail(success_rate_boundary, observation_result, mfpr_rate, metric_name="success"):

    text = f""" The model that generated this 
                                            sample result **may NOT be considered >{success_rate_boundary*100.:0.1f}% {metric_name}**. 😦 
                
**Reason**  
The observed **Sample FPR** is **larger** than the **Risk Factor** 
({observation_result['false_rate'] * 100.:0.2f}%>{mfpr_rate * 100.:0.2f}%). 
        
Passing this model as >{success_rate_boundary*100.:0.1f}% {metric_name} would increase the risk above the committed {mfpr_rate * 100.:0.2f}% (i.e, letting more <{success_rate_boundary*100.:0.1f}% models pass).
        
**Suggested Actions**  
* Explore the reason for the relatively low {metric_name} rate and fix the model.  
* Collect more annotated cases for further justification. Note that the **Risk Factor** must remain at {mfpr_rate * 100.:0.2f}% or lower.  
        """

    return text


def thresh_fail(success_rate_boundary, observed_success_rate, metric_name="success"):
    text = f"""The model that generated the sample sample **may not be considered >{success_rate_boundary*100.:0.1f}% {metric_name}**. 😦   
        
**Reason**  
The Sample Success Rate of {observed_success_rate * 100:0.1f}%  is lower than the threshold of {success_rate_boundary * 100:0.1f}%.  😱 
        
**Suggested Action**  
Explore the reason for the relatively low {metric_name} rate and fix the model. 
        
"""

    return text


def observed_fpr_to_str_fpr_tpr(observed_fpr):
    if observed_fpr < 0.001:
        str_observed_fpr = "<0.1%"
        str_observed_tpr = ">99.9%"
    else:
        str_observed_fpr = f"{observed_fpr * 100:0.1f}%"
        str_observed_tpr = f"{(1.-observed_fpr) * 100:0.1f}%"

    return str_observed_fpr, str_observed_tpr


def results(success_rate_boundary, sample_size, observed_success_rate, observed_fpr, str_over_under, metric_name="success"):
    str_observed_fpr, str_observed_tpr = observed_fpr_to_str_fpr_tpr(observed_fpr)

    text = f"""**Result**   
        An sample of size 
        {sample_size} 
        with a {metric_name} rate of 
        {observed_success_rate * 100:0.1f}% 
        has an **Sample FPR** of 
        {str_observed_fpr}. In other words, there is a {str_observed_tpr}
        probability that this sample was generated by a model with a {metric_name} rate of {str_over_under} 
        {success_rate_boundary * 100:0.1f}%."""

    return text
