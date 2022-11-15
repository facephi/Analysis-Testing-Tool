import streamlit as st
import pandas as pd

# graphs
import altair as alt
import plotly.express as px

# from vega_datasets import data      # csv
import numpy as np
# import glob
import os
#import cv2


@st.cache
def read_csv(file):
    if os.path.splitext(file.name)[1] == '.csv':
        dataframe = pd.read_csv(file)
        return dataframe
    elif os.path.splitext(file.name)[1] == '.json':
        dataframe = pd.read_json(file)
        return dataframe
    else:
        st.markdown("**File not readable!**")


def altair_bar_func(csv_readed):
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


def plotly_bar_func(csv_readed, option0):
    # 3 Columns for placing interactive widgets
    one_column_plotly, two_column_plotly, three_column_plotly = st.columns(
        3)

    with one_column_plotly:
        # Axes
        option_X = st.selectbox(
            'Eje X', csv_readed.head(0).columns)
        option_Y = st.selectbox(
            'Eje Y', csv_readed.head(0).columns)

        # Filter by etiquete
        csv_groupby = np.insert(csv_readed.head(0).columns, 0, 'None')
        option_etiqueta = st.selectbox(
            'Groupby', csv_groupby, key='group_etiqueta')

        # If filter was choosed, especify object
        # Then counts cuantity of X in Y, where all has presents same object filter
        if option_etiqueta != 'None':
            option_atributo = st.selectbox(
                'Choose filter in {}'.format(option_etiqueta), csv_readed[option_etiqueta].unique(), key='group_etiqueta_atribute')

            dif_df = csv_readed[csv_readed[option_etiqueta]
                                == option_atributo]
            df_g = dif_df.groupby(
                [option_X, option_Y]).size().reset_index()
        # Else counts cuantity of X in Y
        else:
            df_g = csv_readed.groupby(
                [option_X, option_Y]).size().reset_index()

        # Change cuantity to percentage
        percentage = st.checkbox('Percentage (%)')

        # Cluster all pipelines results that are not OK or Spoof to FTA
        generateFTA = st.checkbox('Group FTA')

        # Hide FTA in graph so it not infers in percentage calculation (FAR)
        no_fta = st.checkbox('Hide FTA')

        # If path attribute exists, show a button for showing images examples
        if 'path' in csv_readed.columns:
            show_examples = st.checkbox('Show Examples')

    with two_column_plotly:
        # Adjust font type
        font_selector = st.text_input(
            'Font type (arial, verdana, calibri...)', 'calibri black')

        # Order option for showing data in graph
        option_order = st.selectbox('Order', ("trace", "category ascending", "category descending", "array", "total ascending", "total descending", "min ascending", "min descending",
                                              "max ascending", "max descending", "sum ascending",
                                              "sum descending", "mean ascending", "mean descending", "median ascending", "median descending"), index=5)

        # Divides the axis in 2 values, data below that mark (threshold) and data above
        range_selector = st.text_input(
            'Threshold (default None)', 'None')

    with three_column_plotly:
        # Slider to select font size
        option_plt_font_slider = st.slider(
            "FontSize", 10, 30, value=18)

        # Slider to select legend size
        option_plt_legend_font_slider = st.slider(
            "LegendSize", 10, 30, value=16)

        # Slider to select bins value (steps)
        option_plt_slider = st.slider(
            "Bins", 1, 100, value=10)

    # Rename every pipeline result that is not "OK" or "Face liveness - spoof detected" to FTA
    if generateFTA:
        df_g.loc[(df_g[option_X] != "Face liveness - spoof detected") & (df_g[option_X]
                                                                         != "OK"), option_X] = "FTA"
    # Gets every row excepts FTA results in option_X
    if no_fta:
        df_g = df_g.loc[df_g[option_X]
                        != 'FTA', :]

    # If threshold typed, split data in to colums, below and over threshold
    if range_selector != 'None':
        df_g['type'] = df_g[option_X] >= float(range_selector)
        df_g.columns = [option_X, option_Y, 'type', 'Counts']
        df_g['Counts'] = df_g['Counts'].replace(
            {True: "FAR", False: "Accuracy"})
        sorted_df = df_g.sort_values(
            by=[option_Y], ascending=True)

    # Else show normal data
    else:
        df_g.columns = [option_X, option_Y, 'Counts']
        sorted_df = df_g.sort_values(
            by=[option_Y], ascending=True)

    # Theshold and percentage
    if percentage and range_selector != 'None':
        fig = px.histogram(sorted_df, y='type', x=[
            'Counts'], color=option_Y, nbins=option_plt_slider,
            histnorm='percent', text_auto=True,
        )
    # Theshold and not percentage
    elif range_selector != 'None' and not percentage:
        fig = px.histogram(sorted_df, y='type', x=[
            'Counts'], color=option_Y, nbins=option_plt_slider, text_auto=True,
        )
    # Not theshold and percentage
    elif percentage and range_selector == 'None':
        fig = px.histogram(sorted_df, x=option_X, y=[
            'Counts'], color=option_Y, nbins=option_plt_slider, histnorm='percent', text_auto=True,
        )
    # Not theshold and not percentage
    elif not percentage and range_selector == 'None':
        fig = px.histogram(sorted_df, x=option_X, y=[
            'Counts'], color=option_Y, nbins=option_plt_slider, text_auto=True,
        )

    # Parametros generales grafica
    fig.update_layout(barmode='group',
                      xaxis={'categoryorder': option_order},
                      font=dict(family=font_selector,
                                size=option_plt_font_slider),
                      legend=dict(
                          font=dict(size=option_plt_legend_font_slider)),
                      # uniformtext_mode='show',
                      # texttemplate='%{y:.2f}',
                      #   hoverlabel_font=dict(
                      #       size=option_plt_legend_font_slider),
                      yaxis_title="Samples (%)" if percentage else "Samples",
                      title=os.path.splitext(option0.name)[0],
                      # text=[option_Y],
                      )
    # fig.update_xaxes(
    #     tickmode="array",
    #     categoryorder="total ascending",
    #     tickvals=csv_readed[option_Y].unique(),
    #     ticktext=csv_readed[option_Y].unique(),
    #     ticklabelposition="inside",
    #     tickfont=dict(color="black"),
    # )
    # fig.add_trace(
    #     text=sorted_df[option_Y],
    # )
    # fig.update_xaxes(range=[0, 30], visible=False)

    # 2 decimals if percentage marked
    if percentage:
        fig.update_traces(texttemplate='%{y:.2f}')

    # Show graph in streamlit page
    st.plotly_chart(fig, use_container_width=True)

    # Images examples
    if show_examples:
        # Mounts hdd with command in sh (sh allowed to use sudo(required))
        if len(os.listdir('/mnt/p/')) == 0:
            os.system('sudo /home/ahinke/scripts/mount.sh')

        # Filters if option selected
        if option_etiqueta != 'None':
            csv_readed = csv_readed[csv_readed[option_etiqueta]
                                    == option_atributo]

        # Orders rows mayor to minor
        csv_readed = csv_readed.sort_values(
            by=[option_X], ascending=False)

        # Image count tittle
        header_images = st.container()

        # Img containers
        img_col_0, img_col_1, img_col_2, img_col_3, img_col_4 = st.columns(
            5)

        # This variable remeber a value per user session, if its not initialize, start at 0
        session = st.session_state
        if 'N' not in session:
            session.N = 0

        # Buttons to navigate between images
        btn_col_next, btn_col_prev5, more_img_info = st.columns([
            4, 10, 10])
        button_nxt5_img = btn_col_next.button('Next 5 images')
        button_prev5_img = btn_col_next.button('Previous 5 images')
        button_nxt20_img = btn_col_prev5.button('Next 20 images')
        button_reset_img = btn_col_prev5.button('Reset images')
        selectbox_img_info = more_img_info.selectbox(
            'Additional caption information', csv_groupby)

        # Check if action is doable
        if button_nxt5_img:
            if session.N + 5 >= csv_readed.shape[0]:
                st.write("ERROR: Limit exceeded!")
            else:
                session.N += 5
        if button_nxt20_img:
            if session.N + 20 >= csv_readed.shape[0]:
                st.write("ERROR: Limit exceeded!")
            else:
                session.N += 20
        if button_prev5_img:
            if session.N - 5 < 0:
                st.write("ERROR: Limit exceeded!")
            else:
                session.N -= 5
        if button_reset_img:
            session.N = 0

        # Image count tittle
        header_images.header(
            "{}/{} images".format(session.N+5, csv_readed.shape[0]))

        # Img width size
        width_img = 300

        # Shows 5 images in streamlit
        i = 0
        while i < 5:
            # Place image number x in column number x
            with eval('img_col_{}'.format(i)):
                # Reads img path
                var = str(
                    '/mnt/p/' + csv_readed.iloc[i+session.N]['path'])
                var = var.replace('\\', '/')

                # Filter marked but no extra info
                if option_etiqueta != 'None' and selectbox_img_info == 'None':
                    st.image(
                        var,  caption=str(option_X)+': '+str(csv_readed.iloc[i+session.N][option_X]) + ' ' + str(option_etiqueta) + ': ' + str(csv_readed.iloc[i+session.N][option_etiqueta]) + ' ' + str(option_Y)+': ' +
                        str(csv_readed.iloc[i+session.N]
                            [option_Y]),
                        channels='BGR', width=width_img)
                # Filter and extra info marked
                elif option_etiqueta != 'None' and selectbox_img_info != 'None':
                    st.image(
                        var,  caption=str(option_X)+': '+str(csv_readed.iloc[i+session.N][option_X]) + ' ' + str(option_etiqueta) + ': ' + str(csv_readed.iloc[i+session.N][option_etiqueta]) + ' ' + str(option_Y)+': ' +
                        str(csv_readed.iloc[i+session.N]
                            [option_Y]) + ' ' + str(selectbox_img_info)+': '+str(csv_readed.iloc[i+session.N][selectbox_img_info]),
                        channels='BGR', width=width_img)
                # No filter and no extra info
                elif option_etiqueta == 'None' and selectbox_img_info == 'None':
                    st.image(
                        var,  caption=str(option_X)+': '+str(csv_readed.iloc[i+session.N][option_X]) +
                        ' ' + str(option_Y)+': ' +
                        str(csv_readed.iloc[i+session.N]
                            [option_Y]),
                        channels='BGR', width=width_img)
                # No filter but extra info marked
                elif option_etiqueta == 'None' and selectbox_img_info != 'None':
                    st.image(
                        var,  caption=str(option_X)+': '+str(csv_readed.iloc[i+session.N][option_X]) +
                        ' ' + str(option_Y)+': ' +
                        str(csv_readed.iloc[i+session.N]
                            [option_Y]) + ' ' + str(selectbox_img_info)+': '+str(csv_readed.iloc[i+session.N][selectbox_img_info]),
                        channels='BGR', width=width_img)
                i += 1


def plantillas(option0):
    header = st.container()
    csv_information = st.container()
    graphs = st.container()

    with header:
        st.title('Information of {}'.format(option0.name))
        st.text(
            'Here it shows all the information about the data uploaded.')

    with csv_information:
        st.header('CSV container')

        csv_readed = read_csv(option0)
        st.write(csv_readed)

    with graphs:
        st.title('Graphs')
        st.text('Choose the data plot you want to work with')

        st.subheader('Bar Chart')
        alt.themes.enable("streamlit")
        left_column_bar, right_column_bar = st.columns(2)

        with left_column_bar:
            show_but_bar_plt = st.checkbox(
                'Show Plotly Bar')
        # with right_column_bar:
        #     show_but_bar_alt = st.checkbox('Show Altair Bar (OLD)')

        # Altair bar
        # ------------------------Graph 5-----------------------------
        # if show_but_bar_alt:
        #     altair_bar_func(csv_readed)

        # Plotly bar
        # ------------------------Graph 6-----------------------------
        if show_but_bar_plt:
            plotly_bar_func(csv_readed, option0)

        # ------------------------Graph 7-----------------------------
        st.subheader('Pie')
        left_column_pieG, right_column_bar_pieG = st.columns(2)
        with left_column_pieG:
            show_but_pie_plt = st.checkbox('Show Plotly Pie Plot')

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
            with left_column_pl:
                st.write('Eje x')
                option45 = st.selectbox(
                    '', csv_readed.head(0).columns)

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

    # User loads csv/json and storages here
    with st.sidebar:
        csv_files = st.file_uploader("Insert csv", accept_multiple_files=True)

    # checks if files were uploaded, then creates a selectbox for user to select his csv/json to analize
    if csv_files:
        csv_names = np.array([])
        for csv_file in csv_files:
            csv_names = np.append(csv_names, csv_file.name)

        with st.sidebar:
            option_csv = st.selectbox('CSV selector', csv_names)

        for csv_file in csv_files:
            if csv_file.name == option_csv:
                plantillas(csv_file)

    # If no files were uploaded, shows introduction page and some explanation about the files this code can read
    else:
        st.title("Introduction")
        st.markdown("""
                    Welcome to the Analysis Testing Tool, in this website you can create the graph you desire using interactive methods.

                    To start, upload your **CSV** or **JSON** files in the left section of the web.

                    *Note: csv has to be in USA/UK format, example: [https://en.wikipedia.org/wiki/Comma-separated_values#Example](%s)*
                    """ % "https://en.wikipedia.org/wiki/Comma-separated_values#Example")


if __name__ == "__main__":
    prueba_select()
