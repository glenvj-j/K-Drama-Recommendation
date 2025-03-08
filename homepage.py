import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import warnings

warnings.simplefilter(action='ignore')

st.set_page_config(
    page_title="K-Drama Recomendation",
    page_icon="",
    layout="wide"
)


# ------------------------------------------------------------------------------

df = pd.read_csv('kdrama_list.csv', index_col=0)

df['Tags'].fillna('Unknown',inplace=True)

columns_all = [('Genre',0.44),('Main Cast',0.25),('Tags',0.2),('Network',0.01),('Content Rating',0.1)]
all_df = []

vect = CountVectorizer()

# vect = CountVectorizer(tokenizer=lambda x:x.split(', '))
for x in range(len(columns_all)) :
    feature = vect.fit_transform(df[columns_all[x][0]])
    df_vector = pd.DataFrame(data=feature.toarray(),columns=vect.get_feature_names_out())
    df_vector = df_vector * columns_all[x][1]
    df_vector.insert(0, 'Name', df['Name'])
    all_df.append(df_vector)

df_vector = pd.concat(all_df,axis=1)

df_vector = df_vector.loc[:, ~df_vector.columns.duplicated()]

# ------------------------------------------------------------------------------
ratings = []
list_drama_selected = []

# st.markdown("<h1 style='text-align: center;'>K-Drama Recommendation App</h1>", unsafe_allow_html=True)
st.image('image/header.jpg')
st.markdown('''


''')

st.container()
col1s, col2s,col3s = st.columns([10, 1, 10])
with col1s :
    list_drama = (list(df['Name'].sort_values()))
    options = st.multiselect(
        "What are your 3 favorite kdrama",
        list_drama
    )
    # Check if selection exceeds limit
    if len(options) > 3:
        st.warning("⚠️ You can only select up to 3 K-Dramas. Please deselect one.")
    try :
        with st.container() :
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    st.image(df[df['Name']==options[0]]['img url'].iloc[0])
                    score_1 = st.number_input(f"Pick a score for '{options[0]}'",1,10,1)
                    ratings.append(score_1)
                    list_drama_selected.append((df_vector['Name']==options[0]))
                except IndexError:
                    # st.write("Choose 1st Title")
                    print('')

            with col2:
                try:
                    st.image(df[df['Name']==options[1]]['img url'].iloc[0])
                    score_2 = st.number_input(f"Pick a score for '{options[1]}'", 1, 10, 1)
                    ratings.append(score_2)
                    list_drama_selected.append((df_vector['Name']==options[1]))
                except IndexError:
                    # st.write("Choose 2nd Title")
                    print('')

            with col3:
                try:
                    st.image(df[df['Name']==options[2]]['img url'].iloc[0])
                    score_3 = st.number_input(f"Pick a score for '{options[2]}'", 1, 10, 1)
                    ratings.append(score_3)
                    list_drama_selected.append((df_vector['Name']==options[2]))
                except IndexError:
                    # st.write("Choose 3rd Title")
                    print('')
    except IndexError:
        print(" ")

with col2s :
    print("")

with col3s :
    try :
        df_user = df_vector[(df_vector['Name']==options[0]) | (df_vector['Name']==options[1]) | (df_vector['Name']==options[2])]
        item_feature_matrix = df_user.loc[:,df_user.columns[1]:]


        # menggambarkan matrix dengan rating

        item_feature_matrix_with_rating = item_feature_matrix.mul(ratings,axis=0)



        nilai_per_genre = item_feature_matrix_with_rating.sum()
        nilai_total = nilai_per_genre.sum()
        user_feature_vector = nilai_per_genre/nilai_total
        user_feature_vector = user_feature_vector.sort_values(ascending=False).head(10)



        # --------

        df_unwatched_movies = df_vector[~((df_vector['Name']==options[0]) | (df_vector['Name']==options[1]) | (df_vector['Name']==options[2]))]



        item_feature_matrix_unwatched = df_unwatched_movies.loc[:,df_user.columns[1]:]



        # Melakukan perkalian antara item feature matrix unwatched dengan user feature vector
        item_feature_unwatched = item_feature_matrix_unwatched.mul(user_feature_vector)


        scoring = item_feature_unwatched.sum(axis=1)
        scoring = scoring*1000


        df_unwatched_movies_score = df_vector[~((df_vector['Name']==options[0]) | (df_vector['Name']==options[1]) | (df_vector['Name']==options[2]))]

        df_unwatched_movies_score['Movie_Score'] = scoring
        df_unwatched_movies_score_top_5 = df_unwatched_movies_score.sort_values(by='Movie_Score',ascending=False).head(5)[['Name','Movie_Score']]
        df_unwatched_movies_score_top_5 = df_unwatched_movies_score_top_5.merge(df,on='Name',how='left')

        # Initialize session state
        if "score" not in st.session_state:
            st.session_state.score = 0  # Start at index 0

        # Callback functions for buttons
        def decrease_score():
            if st.session_state.score > 0:
                st.session_state.score -= 1

        def increase_score():
            if st.session_state.score < len(df_unwatched_movies_score_top_5) - 1:
                st.session_state.score += 1

        # Loading animation
        with st.spinner("Wait for it..."):
            time.sleep(0.5)

        # Display the selected image
        st.markdown('5 Recomendation K-Drama :')
        colss1 , colss2 = st.columns(2)
        with colss1 :
            
            st.image(df_unwatched_movies_score_top_5['img url'].iloc[st.session_state.score])
        with colss2 :
            st.markdown(f'''
                    <div style="font-weight: 1000; font-size: 30px; max-width: 100%; word-wrap: break-word; white-space: normal;">
                    {df_unwatched_movies_score_top_5['Name'].iloc[st.session_state.score]}<br>
            ''', unsafe_allow_html=True)
            st.markdown(f'''
                <div style="font-weight: 300; font-size: 15px; max-width: 100%; word-wrap: break-word; white-space: normal;">
                    Year : {df_unwatched_movies_score_top_5['Year'].iloc[st.session_state.score]} | {df_unwatched_movies_score_top_5['Episode'].iloc[st.session_state.score]} <br><br>
                    {df_unwatched_movies_score_top_5['Network'].iloc[st.session_state.score]}<br><br>
                    Genre : {df_unwatched_movies_score_top_5['Genre'].iloc[st.session_state.score]}<br><br>
                    Main Cast : {df_unwatched_movies_score_top_5['Main Cast'].iloc[st.session_state.score]}
                </div><br>
            ''', unsafe_allow_html=True)
            print()
            with st.popover("Sinopsis"):
                st.markdown(f'''<div style="font-weight: 300; font-size: 15px; max-width: 100%; word-wrap: break-word; white-space: normal;">
                            {df_unwatched_movies_score_top_5['Sinopsis'].iloc[st.session_state.score]}<br><br>
                    </div>
                ''', unsafe_allow_html=True)

        # Navigation buttons
        # st.markdown(f"<h6 style='text-align: center;'> {st.session_state.score + 1} / 5 </h6>", unsafe_allow_html=True)
        left,middle,right = st.columns(3)
        with middle :
             st.markdown(f"<h6 style='text-align: center;'> {st.session_state.score + 1} / 5 </h6>", unsafe_allow_html=True)

        left.button("Prev", use_container_width=True, on_click=decrease_score)
        right.button("Next", use_container_width=True, on_click=increase_score)

        

    except IndexError:
        print(" ")

st.markdown('GitHub : [Source Code](https://github.com/PurwadhikaDev/AlphaGroup_JC_DS_FT_BDG_05_FinalProject/blob/main/README.md?plain=1)')
