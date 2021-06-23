import matplotlib.pyplot as plt
import streamlit as st

from utils_viz import plot_success_rates_methods, \
    plot_boundary_true_false_positive_rates,  \
    plot_metric_accuracy, plot_success_rates

from utils_stats import accuracy_sample_size, observed_success_correctness, SUCCESS_RATE_BOUNDARY


stage_planning = "Planning"
stage_interpreting = "Interpreting"

audit_stage =  st.sidebar.radio('Audit', [stage_planning, stage_interpreting], index=1)

option_accuracy = f"{audit_stage} Accuracy"
option_clearance = f"{audit_stage} Clearance"

calculator_type = st.sidebar.radio('Calculator', [option_accuracy, option_clearance], index=1)

if option_accuracy == calculator_type:
    if stage_planning == audit_stage:

        f"""
        # Audit Accuracy Planner  
    
        We address the question:
        “What is the minimum **Audit Size** necessary to determine a metric's value?“ 
    
        Two value are required to determine the **Audit Size**:
        * **Accuracy** - The more accurate the result, the larger the sample size required. 
        * **Baseline Success Rate** - The expected metric value. The closer this expected value is to 50% the larger the sample size required.
        """

        planning_plug = "Plug In Values"
        planning_all = "View All Value Results (takes about 10 seconds)"
        explore_mode = st.selectbox('Explore', [planning_plug, planning_all])

        if planning_all == explore_mode:
            plot_metric_accuracy(d_sample_size=10)
            st.pyplot(plt.gcf())

        elif planning_plug == explore_mode:

            """    
            ### Instructions  
    ⬅️ On the left hand panel provide the **Expected Success Rate** and **Goal Accuracy** to find out the minimum **Audit Size**.
    """

            benchmark_percent = st.sidebar.number_input('Expected Success Rate (%)', value=80.,
                                                    min_value=50.,
                                                    max_value=99.)
            benchmark = benchmark_percent/100.

            accuracy_goal_percent = st.sidebar.number_input('Goal Accuracy (%)', value=7., min_value=5.,
                                                 max_value=10.)
            accuracy_goal = accuracy_goal_percent / 100.

            result = accuracy_sample_size(benchmark_success_rate=benchmark, accuracy_goal=accuracy_goal)

            f"""An **Audit Size** of size {result['sample_size']} will result in a measurement of {benchmark_percent}% with {result['accuracy'] * 100.:0.1f}% accuracy."""

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
        * **Baseline Success Rate** - This assumes metrics that are binary in nature (only has value "success", "failure") and follow a Bernoulli distribution.
        """

        with st.beta_expander('Definitions'):
            st.write(str_explanation)

    elif stage_interpreting == audit_stage:


        """TBD"""

elif option_clearance == calculator_type:

    success_rate_boundary_percent = st.sidebar.number_input('Model Min Success Rate',
                                                            value=93.0, min_value=90.,
                                                            max_value=99.)

    success_rate_boundary = success_rate_boundary_percent / 100.

    ci_fraction = 0.95
    observed_success_percent = st.sidebar.number_input('Audit Success Rate (in %)',
                                                       value=98., min_value=0.01,
                                                       max_value=99.)

    observed_success_rate = observed_success_percent / 100.

    sample_size = st.sidebar.number_input('Audit Size', value=100, min_value=100,
                                          max_value=10000)

    mfpr_percent = st.sidebar.number_input(
        'Risk Factor: Max False Positive Rate in (%)', value=1., min_value=0.01,
        max_value=5.)
    mfpr_rate = mfpr_percent / 100.

    less_equal = r"""$$\le$$"""

    if stage_interpreting == audit_stage:
        f"""
        # Audit Result Interpreter  
        
        We address the question: 
        “Given an **Audit Size** with an **Audit Safety Rate**, if I determine this model to be >{success_rate_boundary*100.:0.1f}% safe, how correct (or wrong!) would this decision be?”
        
        
        ### Instructions  
        ⬅️ On the left hand panel provide the **Audit Size**, **Audit Safety Rate** and **Risk Factor** to find out if the model that generated this sample result may be considered >{success_rate_boundary*100.:0.1f}% safe.
        """


        text_short_mfpr = f"""
        *A model passes >{success_rate_boundary*100.:0.1f}% safety if the **Audit FPR**{less_equal}{mfpr_percent:0.2f}%.   
        This guarantees, e.g, that for every 1,000 similar pass decisions, we consider a maximum of
        {mfpr_rate * 1000.:0.0f} incorrect decisions (of {less_equal}{success_rate_boundary*100.:0.1f}% safe) to be acceptable.*
        """

        st.sidebar.write(text_short_mfpr)



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
                audit_result_str = f""" The model that generated this 
                                    audit result **may be considered >{success_rate_boundary*100.:0.1f}% safe**! 🎉🎈🎊 
        
        **Reason**  
        The result indicates that **Audit FPR** is **smaller** than **Max FPR** 
                                    ({observation_result['false_rate']* 100.:0.2f}% 
                                    {less_equal}{mfpr_rate * 100.:0.2f}%). 
                                    """
            else:
                # with a FPR smaller than the maximum

                audit_result_str = f""" The model that generated this 
                                            audit result **may NOT be considered >{success_rate_boundary*100.:0.1f}% safe**. 😦 
                
        **Reason**  
        The result indicates that **Audit FPR** is **larger** than **Max FPR** 
        ({observation_result['false_rate'] * 100.:0.2f}%>{mfpr_rate * 100.:0.2f}%). 
        
        Passing this model as >{success_rate_boundary*100.:0.1f}% safe would risk increasing the FPR to above the committed {mfpr_rate * 100.:0.2f}% (i.e, letting more <{success_rate_boundary*100.:0.1f}% models pass).
        
        **Suggested Actions**  
        * Explore the reason for the relatively low safety rate and fix the model.  
        * Collect more annotated cases for further justification. Note that the Max FPR must remain at {mfpr_rate * 100.:0.2f}% or lower.  
        """
        else:
            str_over_under = """**equal or under**"""
            observed_fpr = observation_result['true_rate']
            audit_result_str = f"""The model that generated the audit sample **may not be considered >{success_rate_boundary*100.:0.1f}% safe**. 😦   
        
        **Reason**  
        The Audit Success Rate of {observed_success_rate * 100:0.1f}%  is lower than the threshold of {success_rate_boundary * 100:0.1f}%.  😱 
        
        **Suggested Action**  
        Explore the reason for the relatively low safety rate and fix the model. 
        
        """



        interpretation_boundary_success_text = f"""  
        **Result**   
        An audit of size 
        {sample_size} 
        with a success rate of 
        {observed_success_rate * 100:0.1f}% 
        has an **Audit FPR** of 
        {observed_fpr * 100:0.1f}%. In other words, there is a  {(1.-observed_fpr) * 100:0.1f}%
        probability that this sample was generated by a model with a success rate of {str_over_under} 
        {success_rate_boundary * 100:0.1f}%.
        
        **Conclusion**   
        {audit_result_str}
        
        
        """

        interpretation_boundary_success_text

        show_visual = st.checkbox('Show visualisation', value=True)

        viz_text = """
        ---  
        Visualisation options
        """

        if show_visual:
            st.sidebar.write(viz_text)
            ac_display = st.sidebar.checkbox('AC 95% Confidence Interval', value=False)
            display_ci = st.sidebar.checkbox('HDI 95% Credible Interval', value=False)

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


