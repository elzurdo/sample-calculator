import matplotlib.pyplot as plt
import streamlit as st

from utils_viz import plot_success_rates_methods, \
    plot_boundary_true_false_positive_rates,  \
    plot_metric_accuracy, plot_success_rates

from utils_stats import accuracy_sample_size, observed_success_correctness, SUCCESS_RATE_BOUNDARY
import utils_text


stage_landing = "learn about these calculators"
stage_planning = "plan an audit budget"
stage_interpreting = "interpret audit results"

audit_stage =  st.sidebar.selectbox('I am looking to', [stage_landing, stage_planning, stage_interpreting], index=2)

option_accuracy = "for Accuracy"
option_clearance = "for Passing a Success Rate"


if stage_landing != audit_stage:

    if stage_planning == audit_stage:
        calculator_types = [option_accuracy]
    elif stage_interpreting == audit_stage:
        calculator_types = [option_clearance]

    calculator_type = st.sidebar.selectbox('Calculator Type', calculator_types, index=0)
else:
    calculator_type = None

    """
    # Audit Calculators
    **Calculate a model's performance based on audit data.**
    
    Use these calculators to:  
    * 🔬 Interpret audit results  
    * 💰 Plan an audit budget- determine an audit size, by exploring trade-offs.
    
   We find that understanding how to draw conclusions from a sample 🔬, assists budgeting for the required 
    sample size 💰.
    
    When interpreting audit results 🔬 these calculators can assess the answers the following: 
    * *"How well does the model **perform**"?*.
    * *"Does the model perform **better than value X**?"*. 
    * *"Is the model **performing better than a previous**?"* - model comparison.
    
    """


    text_difference = """
    The questions above are ranked by their level of complexity in answering.
    The reason for this is that the more moving pieces there are to a calculation the more complex it is to answer.
    
    * **Model performance**  - the most simple as it involves only 2 values: sample size and success rate.
    * **Better than *X*** - involves an additional value *X*. 
    * **Model comparison** - assuming each model has its own sample, this requires: two sample sizes and two success rates.
    
   
    """

    with st.beta_expander('What is the difference?'):
        st.write(text_difference)

    """
    ## Let's Go!
    To start, simply select on the left hand panel ⬅️  what you are interesting in calculating: 
    
    * Audit interpretation 🔬 or budgeting 💰?
    
    After which you will be able to a select model question to answer:  
    * **Performance** or **better than value X**?
    
    🚧🚧🚧 
    Apologies, but the **model comparison** option is not available yet.
    🚧🚧🚧 
    """
    



    text_learn_more = """
    There are two good rules of thumb when relating audit results to the model:
    * The audit result is a **proxy** for that of the model. Not equal.
    * 📐 Accuracy is **expensive**. The finer the measurement, the more it's going to cost (larger sample size). 
    
    Main assumption/limitation:  
    These calculators assume coin flip statistics, known as Binomial Trials (or Bernoulli Trials). 
    In other words - a model that generate binary data, e.g, *success* and *failure* (or *True* and *False*; *Heads* or *Tails*).
    
    """

    with st.beta_expander('Learn More'):
        st.write(text_learn_more)


if option_accuracy == calculator_type:
    if stage_planning == audit_stage:

        f"""
        # Audit Accuracy Planner 📐💰 
    
        **Plan your budget based on the desired accuracy.**   
        
        This calculator addresses the question:  
        “What is the minimum **Audit Size** necessary to determine a **Success Rate** value?“ 
    
        Two values are required to determine the **Audit Size**:
        * **Goal Accuracy** - The more accurate the result, the larger the sample size required. 
        * **Baseline Success Rate** - The expected metric value. The closer this expected value is to 50% the larger the sample size required.
        """

        planning_plug = "Explore one scenario plug in"
        planning_all = "View all scenarios (takes about 10 seconds)"
        explore_mode = st.selectbox('One option or many?', [planning_plug, planning_all])

        if planning_all == explore_mode:
            plot_metric_accuracy(d_sample_size=10)
            st.pyplot(plt.gcf())

        elif planning_plug == explore_mode:

            """    
            ### Instructions  
    ⬅️ On the left hand panel provide the **Baseline Success Rate** expected and **Goal Accuracy** to find out the minimum **Audit Size**.
    """

            benchmark_percent = st.sidebar.number_input('Baseline Success Rate (%)', value=80.,
                                                    min_value=50.,
                                                    max_value=99.)
            benchmark = benchmark_percent/100.

            accuracy_goal_percent = st.sidebar.number_input('Goal Accuracy (%)', value=7., min_value=5.,
                                                 max_value=10.)
            accuracy_goal = accuracy_goal_percent / 100.

            result = accuracy_sample_size(benchmark_success_rate=benchmark, accuracy_goal=accuracy_goal)

            f"""### Result   
An minimum **Audit Size** of {result['sample_size']} is required to measure a **Success Rate** of {benchmark_percent}% within {result['accuracy'] * 100.:0.1f}% **Accuracy** (95% CI)."""

            success = result['sample_size'] * benchmark
            failure = result['sample_size'] - success
            plot_success_rates(success, failure,
                               min_psuccess=benchmark - result['accuracy'], max_psucess=benchmark + result['accuracy'], d_psuccess=0.0001,
                               color="purple", format='-', label=None, fill=False,
                               display_ci=True,
                               alpha=1., factor=1., ci_label=None,
                               xlabel="success rate",
                               ylabel="probability distribution function")

            st.pyplot(plt.gcf())




        str_explanation = f"""
        * **Accuracy** - The 95% Credible Interval using the High Density Interval method.
        * **Success Rate** - This assumes metrics that are binary in nature (only has value "success", "failure") and follow a Bernoulli distribution.
        """

        with st.beta_expander('Definitions'):
            st.write(str_explanation)

    elif stage_interpreting == audit_stage:


        """TBD"""

elif option_clearance == calculator_type:

    success_rate_boundary_percent = st.sidebar.number_input('Model Min Success Rate (%)',
                                                            value=93.0, min_value=90.,
                                                            max_value=99.)

    success_rate_boundary = success_rate_boundary_percent / 100.

    ci_fraction = 0.95
    observed_success_percent = st.sidebar.number_input('Audit Success Rate (%)',
                                                       value=98., min_value=0.01,
                                                       max_value=99.)

    observed_success_rate = observed_success_percent / 100.

    sample_size = st.sidebar.number_input('Audit Size', value=100, min_value=100,
                                          max_value=10000)

    mfpr_percent = st.sidebar.number_input(
        'Risk Factor: Max False Positive Rate in (%)', value=1., min_value=0.01,
        max_value=5.)
    mfpr_rate = mfpr_percent / 100.


    if stage_interpreting == audit_stage:
        st.write(utils_text.interpret_pass_header(success_rate_boundary))
        st.sidebar.write(utils_text.risk_factor_explanation(success_rate_boundary, mfpr_rate))



        observation_result = observed_success_correctness(observed_success_rate, sample_size, generator_success_rates=None,
                                         success_rate_boundary=success_rate_boundary,
                                         tpr_method="flat_min", min_ab=0.5)

        boundary_success_bool =  observation_result['rate_type'] == 'success'

        if boundary_success_bool:
            str_over_under = """**over**"""
            observed_fpr = observation_result['false_rate']
            # audit passes boundary
            mfpr_success_bool =  observation_result['false_rate'] <= mfpr_rate

            if mfpr_success_bool:
                # with a FPR smaller than the maximum
                audit_result_str = utils_text.risk_success(success_rate_boundary, observation_result, mfpr_rate)
            else:
                # with a FPR smaller than the maximum

                audit_result_str = utils_text.risk_fail(success_rate_boundary, observation_result, mfpr_rate)
        else:
            str_over_under = """**equal or under**"""
            observed_fpr = observation_result['true_rate']
            audit_result_str = utils_text.thresh_fail(success_rate_boundary, observed_success_rate)

        text_result = utils_text.results(success_rate_boundary, sample_size, observed_success_rate, observed_fpr, str_over_under)


        interpretation_boundary_success_text = \
            f"""### Result Interpretation 
{text_result}   

**Conclusion**  
{audit_result_str}  """

        interpretation_boundary_success_text

        show_visual = st.checkbox('Show visualisation', value=True)

        viz_text = """
        ---  
        Visualisation options
        """

        if show_visual:
            st.sidebar.write(viz_text)
            ac_display = st.sidebar.checkbox('Agresti-Coull 95% CI', value=False)
            display_ci = st.sidebar.checkbox('High Density 95% Credible Interval', value=False)

            plot_boundary_true_false_positive_rates(observed_success_rate, sample_size, success_rate_boundary=success_rate_boundary)
            plot_success_rates_methods(observed_success_rate, sample_size, ci_fraction, ci_type="HDI", legend_title= None, ac_display=ac_display, display_ci=display_ci)
            st.pyplot(plt.gcf())

    elif stage_planning == audit_stage:

        """TBD"""


    text_expanded_mfpr = f"""
    By committing to a **Max False Positive Rate (Max FPR)** of 
    {mfpr_percent}%, results with with a higher **Audit FPR** will not be considered >{success_rate_boundary*100.:0.1f}% safe and hence fail. 
    
    This ensures that that we mitigate the risk at accepting models with relatively lower safety rates than {success_rate_boundary*100.:0.1f}%.
    
    E.g, this commitment guarantees that for every 1,000 audits, we should expect and average of 
    {mfpr_rate * 1000.:0.1f} to be False Positives.
    """


    with st.beta_expander("""Risk mitigation by Max FPR"""):
        st.write(text_expanded_mfpr)


"""
Created by: [Eyal Kazin](https://www.linkedin.com/in/eyal-kazin-0b96227a/)  
"""


