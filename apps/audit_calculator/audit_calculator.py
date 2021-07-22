import matplotlib.pyplot as plt
import streamlit as st

from utils_viz import plot_success_rates_methods, \
    plot_boundary_true_false_positive_rates,  \
    plot_metric_accuracy, plot_success_rates

from utils_stats import accuracy_sample_size, hdi_ci_limits, observed_success_correctness, SUCCESS_RATE_BOUNDARY, sample_rate_fpr_to_size, CI_FRACTION
import utils_text

MIN_AUDIT_SIZE = 20
MIN_AUDIT_SIZE_CLEARANCE = 50

image_differences = "https://user-images.githubusercontent.com/6064016/123537964-3bfd8200-d72a-11eb-8431-47ac1530b193.png"

stage_landing = "💡 learn about these calculators."
stage_planning = "💰 plan an audit budget"
stage_interpreting = "🔬 interpret results"


st.sidebar.write("""## Select a Calculator""")
audit_stage =  st.sidebar.selectbox('I would like to ...', [stage_landing, stage_planning, stage_interpreting], index=0)

option_accuracy = "📐 determine accuracy."
option_clearance = "⛑ ensure value clearance."

if stage_landing != audit_stage:
    if stage_planning == audit_stage:
        calculator_types = [option_accuracy, option_clearance]
        calc_index = 0
    elif stage_interpreting == audit_stage:
        calculator_types = [option_accuracy, option_clearance]
        calc_index = 0

    calculator_type = st.sidebar.selectbox('in order to ...', calculator_types, index=calc_index)

    st.sidebar.write("""---""")

    metric_name = st.sidebar.text_input('Parameter name', value='success').lower()
    metric_name_title = metric_name.title()

    st.sidebar.write("""## Calculator Parameters""")
else:
    calculator_type = None
    st.write(utils_text.landing_header())

    st.image(image_differences, caption="Visualising the three types of calculators.  The bell shapes represent the model probability given audit data (known as the posterior distribution).")


    text_difference = """
    The questions above have different levels of difficulty, because the more moving pieces there are to a calculation the more complex it is to answer.
    
    * **Determining Accuracy**  - the most simple involving only: sample size and success rate.
    * **Ensuring Value Clearance** - sample size, success rate the said value. 
    * **Model Comparison** - each model has its own sample, i.e: 2 sample sizes and 2 success rates.
    """

    with st.beta_expander('Why three separate calculations?'):
        st.write(text_difference)

    st.write(utils_text.landing_instructions())

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
        st.write(utils_text.plan_accuracy_header(metric_name=metric_name))

        planning_plug = "Explore one scenario plug in"
        planning_all = "View all scenarios (takes about 10 seconds)"
        explore_mode = st.selectbox('One option or many?', [planning_plug, planning_all])

        if planning_all == explore_mode:
            plot_metric_accuracy(d_sample_size=10)
            st.pyplot(plt.gcf())

        elif planning_plug == explore_mode:

            f"""    
            ### Instructions  
    ⬅️ On the left hand panel provide the **Baseline {metric_name_title} Rate** expected and **Goal Accuracy** to find out the minimum **Audit Size**.
    """

            benchmark_percent = st.sidebar.number_input(f'Baseline {metric_name_title} Rate (%)', value=80.,
                                                    min_value=1.,
                                                    max_value=99.)
            benchmark = benchmark_percent/100.

            accuracy_goal_percent = st.sidebar.number_input('Goal Accuracy (%)', value=7., min_value=5.,
                                                 max_value=10.)
            accuracy_goal = accuracy_goal_percent / 100.

            result = accuracy_sample_size(benchmark_success_rate=benchmark, accuracy_goal=accuracy_goal)

            f"""### Result   
The minimum **Audit Size** required to measure a **{metric_name_title} Rate** of {benchmark_percent}% within {result['accuracy'] * 100.:0.1f}% **Accuracy** (95% credible interval) is:  
# {result['sample_size']} samples 
"""

            success = result['sample_size'] * benchmark
            failure = result['sample_size'] - success
            plot_success_rates(success, failure,
                               min_psuccess=benchmark - result['accuracy'], max_psucess=benchmark + result['accuracy'], d_psuccess=0.0001,
                               color="purple", format='-', label=None, fill=False,
                               display_ci=True,
                               alpha=1., factor=1., ci_label=None,
                               xlabel=f"{metric_name} rate",
                               ylabel="probability distribution function")

            st.pyplot(plt.gcf())


    elif stage_interpreting == audit_stage:
        #st.write(utils_text.plan_accuracy_header(metric_name=metric_name))
        observed_success_percent = st.sidebar.number_input(
            f'Audit {metric_name_title} Rate (%)', value=80.,
            min_value=1.,
            max_value=99.)
        observed_success_rate = observed_success_percent / 100.

        sample_size = st.sidebar.number_input('Audit Size', value=100, min_value=MIN_AUDIT_SIZE,
                                              max_value=10000)

        st.write(utils_text.interpret_accuracy_header(CI_FRACTION, metric_name=metric_name))

        ci_min, ci_max = hdi_ci_limits(observed_success_rate, sample_size, ci_fraction=CI_FRACTION)

        f"""### Result   
An audit size of {sample_size:,} with a **{metric_name_title} Rate** of {observed_success_rate * 100.:0.1f}% was most likely to be generated by a model with a {metric_name} rate:
# between {ci_min * 100:0.1f}% and {ci_max * 100:0.1f}%
({CI_FRACTION * 100.:0.1f}% credible interval of with **{(ci_max - ci_min) * 100:0.1f}%**)
"""

        ci_width = ci_max - ci_min

        success = sample_size * observed_success_rate
        failure = sample_size - success

        xmin = ci_min - ci_width
        xmax = ci_max + ci_width

        plot_success_rates(success, failure,
                           min_psuccess=xmin,
                           max_psucess=xmax,
                           xmin=xmin, xmax=xmax,
                           d_psuccess=0.0001,
                           color="purple", format='-', label=None, fill=False,
                           display_ci=True,
                           alpha=1., factor=1., ci_label=None,
                           xlabel=f"{metric_name} rate",
                           ylabel="probability distribution function")

        st.pyplot(plt.gcf())

    str_explanation = f"""
    * **Accuracy** - The 95% Credible Interval using the High Density Interval method.
    * **{metric_name_title} Rate** - This assumes metrics that are binary in nature (only has value "success", "failure") and follow a Bernoulli distribution.
    """

    with st.beta_expander('Definitions'):
        st.write(str_explanation)

elif option_clearance == calculator_type:

    success_rate_boundary_percent = st.sidebar.number_input(f'Model min {metric_name.title()} Rate (%)',
                                                            value=93.0, min_value=1.,
                                                            max_value=99.)

    success_rate_boundary = success_rate_boundary_percent / 100.

    min_observed_success_percent = 0.01

    if stage_planning == audit_stage:
        min_observed_success_percent = success_rate_boundary_percent + 0.01

    observed_success_percent = st.sidebar.number_input(f'Audit {metric_name.title()} Rate (%)',
                                                       value=98., min_value=min_observed_success_percent,
                                                       max_value=99.)

    observed_success_rate = observed_success_percent / 100.

    if audit_stage in [stage_interpreting]:
        sample_size = st.sidebar.number_input('Audit Size', value=100, min_value=MIN_AUDIT_SIZE_CLEARANCE,
                                              max_value=10000)

    mfpr_percent = st.sidebar.number_input(
        'Risk Factor: max False Positive Rate (%)', value=1., min_value=0.01,
        max_value=5.)
    mfpr_rate = mfpr_percent / 100.

    st.sidebar.write(
        utils_text.risk_factor_explanation(success_rate_boundary, mfpr_rate, metric_name=metric_name))

    if stage_interpreting == audit_stage:
        st.write(utils_text.interpret_pass_header(success_rate_boundary, metric_name=metric_name))
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
                audit_result_str = utils_text.risk_success(success_rate_boundary, observation_result, mfpr_rate, metric_name=metric_name)
            else:
                # with a FPR smaller than the maximum

                audit_result_str = utils_text.risk_fail(success_rate_boundary, observation_result, mfpr_rate, metric_name=metric_name)
        else:
            str_over_under = """**equal or under**"""
            observed_fpr = observation_result['true_rate']
            audit_result_str = utils_text.thresh_fail(success_rate_boundary, observed_success_rate, metric_name=metric_name)

        text_result = utils_text.results(success_rate_boundary, sample_size, observed_success_rate, observed_fpr, str_over_under, metric_name=metric_name)


        interpretation_boundary_success_text = \
            f"""### Result Interpretation 
{text_result}   

**Conclusion**  
{audit_result_str}  """

        interpretation_boundary_success_text

    elif stage_planning == audit_stage:
        st.write(utils_text.plan_clearance_header(success_rate_boundary, metric_name=metric_name))

        max_sample_size = 1000

        sample_size = sample_rate_fpr_to_size(observed_success_rate, mfpr_rate, success_rate_boundary=success_rate_boundary, max_sample_size=max_sample_size)

        this_text = \
            f"""### Results 
To ensure a model {metric_name} rate is larger than {success_rate_boundary_percent}% with a **Risk Factor** of {mfpr_percent}% under the condition that 
the observable **Audit {metric_name_title} Rate** is at least {observed_success_percent}% requires"""

        if sample_size is not None:
            this_text += f""" a sample of size: 
# {int(sample_size):,} samples
"""

        else:
            this_text += f""" a sample with more than {max_sample_size:,} samples. 

Please adjust either the **Risk Factor** or the **Audit {metric_name_title} Rate.**"""
            sample_size = max_sample_size

        this_text

    show_visual = st.checkbox('Show visualisation', value=True)

    if show_visual:
        viz_text = """
        ---  
        ## Visualisation options
        """
        st.sidebar.write(viz_text)
        ac_display = st.sidebar.checkbox('Agresti-Coull 95% CI', value=False)
        display_ci = st.sidebar.checkbox('High Density 95% Credible Interval', value=False)

        plot_boundary_true_false_positive_rates(observed_success_rate, sample_size, success_rate_boundary=success_rate_boundary)
        plot_success_rates_methods(observed_success_rate, sample_size, CI_FRACTION, ci_type="HDI", legend_title= None, ac_display=ac_display, display_ci=display_ci, metric_name=metric_name, xmin=success_rate_boundary - 0.05)
        st.pyplot(plt.gcf())


    text_expanded_mfpr = f"""
    By committing to a **Max False Positive Rate (Max FPR)** of 
    {mfpr_percent}%, results with with a higher **Audit FPR** will not be considered >{success_rate_boundary*100.:0.1f}% {metric_name} and hence fail. 
    
    This ensures that that we mitigate the risk at accepting models with relatively lower {metric_name} rates than {success_rate_boundary*100.:0.1f}%.
    
    E.g, this commitment guarantees that for every 1,000 audits, we should expect and average of 
    {mfpr_rate * 1000.:0.1f} to be False Positives.
    """


    with st.beta_expander("""Importance of the Risk Factor"""):
        st.write(text_expanded_mfpr)


"""
Created by: [Eyal Kazin](https://www.linkedin.com/in/eyal-kazin-0b96227a/)  
"""


