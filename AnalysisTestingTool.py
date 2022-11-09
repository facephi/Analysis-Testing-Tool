import streamlit as st
import pandas as pd

# graphs
import altair as alt
import plotly.express as px

from vega_datasets import data      # csv
import numpy as np
import glob
import os


@st.cache
def read_csv(file):
    csv = pd.read_csv(file)

    return csv


# @st.cache
# # @experimental_memo
# def read_csv_2(name):
#     csv = pd.read_csv('data/iris.csv')
#     return csv

# def f(row):
#     if row['pipeline_result'] == 'Face liveness - spoof detected':
#         val = 'FA'
#     elif row['pipeline_result'] == 'OK':
#         val = 'OK'
#     else:
#         val = 'FTA'
#     return val


def plantillas(option0):
    header = st.container()
    csv_information = st.container()
    graphs = st.container()

    with header:
        st.title('Information of {}'.format(option0.name))
        st.text(
            'Here we have all the information about the data uploaded.')
        # st.subheader('DataBase Selector')
        # option1 = st.selectbox('', ['iris', 'cars'], key='0')

    with csv_information:
        # st.header('Database Used')
        # st.text('Information about testing method')

        st.header('CSV container')

        csv_readed = read_csv(option0)
        st.write(csv_readed)

    with graphs:
        st.header('Graphs')
        st.text('Choose the data plot you want to work with')

        #difDf = pd.DataFrame()

        # dif_df = csv_readed[csv_readed.Emisor == 'EM01']
        # st.write(dif_df)

        # Altair plot
        # ------------------------Graph 5-----------------------------
        st.subheader('Bar Chart')
        alt.themes.enable("streamlit")
        left_column_bar, right_column_bar = st.columns(2)
        # You can use a column just like st.sidebar:
        with left_column_bar:
            show_but_bar_plt = st.checkbox(
                'Show Plotly Bar')
        with right_column_bar:
            show_but_bar_alt = st.checkbox('Show Altair Bar (OLD)')

        if show_but_bar_alt:
            selection = alt.selection_multi(
                fields=['database'], bind='legend')
            option4 = st.selectbox('', csv_readed.head(0).columns)
            # Add a selectbox to the sidebar:
            left_column, right_column = st.columns(2)
            # You can use a column just like st.sidebar:
            with left_column:
                agree = st.checkbox('Color')
                percentage_alt = st.checkbox('Percentage')
                buttonstr = st.checkbox('X is string')

            # Or even better, call Streamlit functions inside a "with" block:
            with right_column:
                option_color = st.selectbox(
                    '', csv_readed.head(0).columns, key='8')

            option4_s = st.slider("", 1, 800, key='6')

            if percentage_alt:
                chart_layer = (
                    alt.Chart(csv_readed).transform_joinaggregate(
                        total='count(*)'
                    ).transform_calculate(
                        pct='1 / datum.total'
                    ).mark_bar().encode(
                        alt.X(option4, bin=alt.BinParams(step=option4_s/10)
                              ) if not buttonstr else alt.X(option4),
                        alt.Y('sum(pct):Q', axis=alt.Axis(format='%')),
                        alt.Color(option_color
                                  ) if agree else alt.Color(),
                        opacity=alt.condition(
                            selection, alt.value(1.0), alt.value(0.2))
                    )
                    .add_selection(selection)
                    .interactive()
                )
            else:
                chart_layer = (
                    alt.Chart(csv_readed).mark_bar().encode(
                        alt.X(option4, bin=alt.BinParams(step=option4_s/10)
                              ) if not buttonstr else alt.X(option4),
                        alt.Y(aggregate='count', type='quantitative'),
                        alt.Color(option_color
                                  ) if agree else alt.Color(),
                        opacity=alt.condition(
                            selection, alt.value(1.0), alt.value(0.2))
                    )
                    .add_selection(selection)
                    .interactive()
                )

            st.altair_chart(chart_layer, use_container_width=True)

        # ------------------------Graph 6-----------------------------
        if show_but_bar_plt:
            one_column_plotly, two_column_plotly, three_column_plotly = st.columns(
                3)
            # You can use a column just like st.sidebar:
            with one_column_plotly:
                option44 = st.selectbox(
                    'Eje X', csv_readed.head(0).columns)
                optionplt_color = st.selectbox(
                    'Eje Y', csv_readed.head(0).columns)

                csv_groupby = np.insert(csv_readed.head(0).columns, 0, 'None')
                option_etiqueta = st.selectbox(
                    'Groupby', csv_groupby, key='group_etiqueta')

                if option_etiqueta != 'None':
                    option_atributo = st.selectbox(
                        'Choose filter in {}'.format(option_etiqueta), csv_readed[option_etiqueta].unique(), key='group_etiqueta_atribute')

                    dif_df = csv_readed[csv_readed[option_etiqueta]
                                        == option_atributo]
                    st.write(dif_df)
                    df_g = dif_df.groupby(
                        [option44, optionplt_color]).size().reset_index()
                else:
                    df_g = csv_readed.groupby(
                        [option44, optionplt_color]).size().reset_index()

                percentage = st.checkbox('Percentage (%)')
                generateFTA = st.checkbox('Group FTA')
                no_fta = st.checkbox('Hide FTA')

            with two_column_plotly:
                font_selector = st.text_input(
                    'Font type (arial, verdana, calibri...)', 'calibri black')
                option_order = st.selectbox('Order', ("trace", "category ascending", "category descending", "array", "total ascending", "total descending", "min ascending", "min descending",
                                                      "max ascending", "max descending", "sum ascending",
                                                      "sum descending", "mean ascending", "mean descending", "median ascending", "median descending"), index=5)
                range_selector = st.text_input(
                    'Threshold (default None)', 'None')
                # facephi = st.checkbox('FacePhi_technology')
                # idrnd = st.checkbox('IDR&D_technology')

            with three_column_plotly:

                option_plt_font_slider = st.slider(
                    "FontSize", 10, 30, value=18)
                option_plt_legend_font_slider = st.slider(
                    "LegendSize", 10, 30, value=16)
                option_plt_slider = st.slider(
                    "Bins", 1, 100, value=10)
            # ------------------------Graph 6.1-----------------------------
            if generateFTA:
                df_g.loc[(df_g[option44] != "Face liveness - spoof detected") & (df_g[option44]
                         != "OK"), option44] = "FTA"
                df_g.loc[df_g[option44] ==
                         "Face liveness - spoof detected", option44] = "Accuracy"
                df_g.loc[df_g[option44] == "OK", option44] = "FAR"
                # st.write(df_g)
            if no_fta:
                df_g = df_g.loc[df_g[option44]
                                != 'FTA', :]

            # if facephi:
            #     df_g['Technology'] = 'Facephi'

            # if idrnd:
            #     df_g['Technology'] = 'IDRND'

            # Si se marca el box porcentaje
            if range_selector != 'None':
                df_g['type'] = df_g[option44] >= float(range_selector)
                df_g.columns = [option44, optionplt_color, 'type', 'Counts']
                # df_g['Counts'] = df_g['Counts'].replace({True: "{} por encima o igual de {}".format(option44,
                #                                                                                     range_selector), False: "{} por debajo de {}".format(option44, range_selector)})
                df_g['Counts'] = df_g['Counts'].replace(
                    {True: "FAR", False: "Accuracy"})

                sorted_df = df_g.sort_values(
                    by=[optionplt_color], ascending=True)
                if percentage:
                    fig = px.histogram(sorted_df, y='type', x=[
                        'Counts'], color=optionplt_color, nbins=option_plt_slider, histnorm='percent', text_auto=True)
                else:
                    fig = px.histogram(sorted_df, y='type', x=[
                        'Counts'], color=optionplt_color, nbins=option_plt_slider, text_auto=True)
            else:
                df_g.columns = [option44, optionplt_color, 'Counts']
                sorted_df = df_g.sort_values(
                    by=[optionplt_color], ascending=True)
                if percentage:
                    fig = px.histogram(sorted_df, x=option44, y=[
                        'Counts'], color=optionplt_color, nbins=option_plt_slider, histnorm='percent', text_auto=True)
                else:
                    fig = px.histogram(sorted_df, x=option44, y=[
                        'Counts'], color=optionplt_color, nbins=option_plt_slider, text_auto=True)

            # Parametros generales grafica
            fig.update_layout(barmode='group',                              # ,xaxis={'categoryorder': 'max ascending'}
                              # uniformtext_minsize=12,
                              xaxis={'categoryorder': option_order},
                              font=dict(family=font_selector,
                                        size=option_plt_font_slider),
                              legend=dict(
                                  font=dict(size=option_plt_legend_font_slider)),
                              # uniformtext_mode='show',
                              # texttemplate='%{y:.2f}',
                              #   title=dict(
                              #       font=dict(size=option_plt_legend_font_slider)),
                              #   hoverlabel_font=dict(
                              #       size=option_plt_legend_font_slider),
                              yaxis_title="Samples (%)" if percentage else "Samples",
                              title=os.path.splitext(option0.name)[0]
                              )
            if percentage:
                fig.update_traces(texttemplate='%{y:.2f}')

            # Mostrar grafica en streamlit
            st.plotly_chart(fig, use_container_width=True)

        # ------------------------Graph 7-----------------------------
        st.subheader('Pie')
        left_column_pieG, right_column_bar_pieG = st.columns(2)
        # You can use a column just like st.sidebar:
        with left_column_pieG:
            show_but_pie_plt = st.checkbox('Show Plotly Pie Plot')
        with right_column_bar_pieG:
            show_but_pie_alt = st.checkbox(
                'Show Altair Pie Plot (OLD)')

        if show_but_pie_alt:
            selection = alt.selection_multi(fields=['species'], bind='legend')
            left_column_pie, right_column_pie = st.columns(2)
            # You can use a column just like st.sidebar:
            with left_column_pie:
                option_pie = st.selectbox(
                    '', csv_readed.head(0).columns)

            # Or even better, call Streamlit functions inside a "with" block:
            with right_column_pie:
                option_color_pie = st.selectbox(
                    '', csv_readed.head(0).columns)

            chart_layer = (
                alt.Chart(csv_readed).transform_joinaggregate(
                    total='count(*)'
                ).transform_calculate(
                    pct='1 / datum.total'
                ).mark_arc().encode(
                    theta=alt.Theta(field=option_pie, type="quantitative"),
                    color=alt.Color(field=option_color_pie, type="nominal"),
                    opacity=alt.condition(
                        selection, alt.value(1.0), alt.value(0.2))
                )
                .add_selection(selection)
            )

            st.altair_chart(chart_layer, use_container_width=True)

        # ------------------------Graph 7-----------------------------
        if show_but_pie_plt:
            option_pie_plotly = st.selectbox(
                '', csv_readed.head(0).columns)
            new_df = csv_readed[option_pie_plotly].value_counts().rename_axis(
                option_pie_plotly).reset_index(name='counts')
            fig = px.pie(new_df, values='counts', names=option_pie_plotly)
            st.plotly_chart(fig, use_container_width=True)

        # ------------------------Graph 8-----------------------------
        st.subheader('Heatmap')
        show_but_bar_plt2 = st.checkbox('Show Plotly Heatmap')
        if show_but_bar_plt2:
            left_column_pl, right_column_pl = st.columns(2)
            # You can use a column just like st.sidebar:
            with left_column_pl:
                st.write('Eje x')
                option45 = st.selectbox(
                    '', csv_readed.head(0).columns)

            # Or even better, call Streamlit functions inside a "with" block:
            with right_column_pl:
                st.write('Eje y')
                option46 = st.selectbox(
                    '', csv_readed.head(0).columns)

            fig = px.density_heatmap(
                csv_readed, x=option45, y=option46, color_continuous_scale="Viridis")

            # fig = px.bar(csv_readed, x=option45, y=option45.value_counts(),
            #              color="species" if agree else None,
            #              title="Plotly")

            st.plotly_chart(fig, use_container_width=True)

        # ------------------------Graph 8-----------------------------
        st.subheader('Bubble')
        show_but_bar_plt2 = st.checkbox('Show Plotly Bubble')
        if show_but_bar_plt2:
            left_column_pl, right_column_pl, other_column, other_column2 = st.columns(
                4)
            # You can use a column just like st.sidebar:
            with left_column_pl:
                st.write('Eje x')
                option47 = st.selectbox(
                    '', csv_readed.head(0).columns)

            # Or even better, call Streamlit functions inside a "with" block:
            with right_column_pl:
                st.write('Eje y')
                option48 = st.selectbox(
                    '', csv_readed.head(0).columns)
            # list = csv_readed.head(0).columns
            # list.append('None')
            with other_column:
                st.write('Size')
                list = np.array(csv_readed.head(0).columns)
                list = np.append(list, 'None')
                option49 = st.selectbox(
                    '', list, key='49')
            with other_column2:
                st.write('Color')
                option50 = st.selectbox(
                    '', csv_readed.head(0).columns)

            fig = px.scatter(csv_readed, x=option47, y=option48,
                             size=option49 if option49 != 'None' else None,
                             color=option50,
                             # hover_name="country",
                             log_x=True, size_max=30)

            st.plotly_chart(fig, use_container_width=True)

        # ------------------------Graph 9-----------------------------
        st.subheader('3D')
        show_but_bar_plt2 = st.checkbox('Show Plotly 3D')
        if show_but_bar_plt2:
            left_column_pl, right_column_pl, other_column, other_column2 = st.columns(
                4)
            # You can use a column just like st.sidebar:
            with left_column_pl:
                st.write('Eje x')
                option51 = st.selectbox(
                    '', csv_readed.head(0).columns)

            # Or even better, call Streamlit functions inside a "with" block:
            with right_column_pl:
                st.write('Eje y')
                option52 = st.selectbox(
                    '', csv_readed.head(0).columns)
            with other_column:
                st.write('Eje z')
                option53 = st.selectbox(
                    '', csv_readed.head(0).columns)
            with other_column2:
                st.write('Color')
                option54 = st.selectbox(
                    '', csv_readed.head(0).columns)

            fig = px.scatter_3d(csv_readed, x=option51, y=option52, z=option53,
                                color=option54)

            st.plotly_chart(fig, use_container_width=True)


def prueba_select():
    # Page initial configuration
    st.set_page_config(page_title="Analysis Testing Tool",
                       layout="wide",
                       initial_sidebar_state="expanded")

    # Adding Title to the page
    cols = st.columns([4, 3, 4])
    cols[0].image("data/src/logo-facephi.png",
                  output_format='PNG')
    cols[2].title("Analysis Testing Tool")
    with st.sidebar:
        csv_files = st.file_uploader("Insert csv", accept_multiple_files=True)

    if csv_files:
        csv_names = np.array([])
        for csv_file in csv_files:
            csv_names = np.append(csv_names, csv_file.name)

        with st.sidebar:
            option_csv = st.selectbox('CSV selector', csv_names)

        for csv_file in csv_files:
            if csv_file.name == option_csv:
                plantillas(csv_file)
    else:
        st.title("Introduction")
        st.markdown("""
                    Welcome to the Analysis Testing Tool, in this website you can create the graph you desire using interactive methods.
                    
                    To start, upload your csv files in the left section of the web. 
                    
                    *Note: csv has to be in USA/UK format, example: [https://en.wikipedia.org/wiki/Comma-separated_values#Example](%s)*
                    """ % "https://en.wikipedia.org/wiki/Comma-separated_values#Example")
        # st.write("Bienvenido al creador de gráficas, para comenzar introduce tus datos en formato csv en el desplegable de la izquierda")

    # if csv_files:
    #     plantillas(csv)

    # if option0 == 'DPAD_Screen':
    #     files2 = []
    #     [files2.extend(glob.glob('data/DPAD_Screen/**/*' + e, recursive=True))
    #      for e in '.csv']
    #     with st.sidebar:
    #         option_prueba = st.selectbox(
    #             'Prueba Selector', files2, key='90')
    #     #csv_readed = read_csv(option_prueba)
    #     plantillas(option_prueba)
    # elif option0 == 'Eyes Closed/Open':
    #     files2 = []
    #     [files2.extend(glob.glob('data/IDNRND_closed_open/**/*' + e, recursive=True))
    #      for e in '.csv']
    #     with st.sidebar:
    #         option_prueba = st.selectbox(
    #             'Prueba Selector', files2, key='90')
    #     #csv_readed = read_csv(option_prueba)
    #     plantillas(option_prueba)
    # else:
    #     #csv_readed = read_csv(option0)
    #     optionpath = 'data/csv_files/' + option0 + '.csv'
    #     plantillas(optionpath)


if __name__ == "__main__":
    prueba_select()
