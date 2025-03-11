import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import warnings

warnings.simplefilter(action='ignore')

st.set_page_config(
    page_title="K-Drama Recommendation",
    page_icon="",
    layout="wide"
)

# ------------------------------------------------------------------------------

df = pd.read_csv('kdrama_list.csv', index_col=0)
df['Tags'].fillna('Unknown', inplace=True)

columns_all = [('Genre', 0.44), ('Main Cast', 0.25), ('Tags', 0.2), ('Network', 0.01), ('Content Rating', 0.1)]
all_df = []

vect = CountVectorizer()

for x in range(len(columns_all)):
    feature = vect.fit_transform(df[columns_all[x][0]])
    df_vector = pd.DataFrame(data=feature.toarray(), columns=vect.get_feature_names_out())
    df_vector = df_vector * columns_all[x][1]
    df_vector.insert(0, 'Name', df['Name'])
    all_df.append(df_vector)

df_vector = pd.concat(all_df, axis=1)
df_vector = df_vector.loc[:, ~df_vector.columns.duplicated()]

# ------------------------------------------------------------------------------

st.image('image/header.jpg')

st.container()
col1s, col2s, col3s = st.columns([10, 1, 10])
ratings = []
list_drama_selected = []

with col1s:
    list_drama = list(df['Name'].sort_values())
    options = st.multiselect("What are your 3 favorite K-Dramas", list_drama)

    # Ensure max 3 selections
    if len(options) > 3:
        st.warning("⚠️ You can only select up to 3 K-Dramas. Please deselect one.")

    if len(options) > 0:
        try:
            col1, col2, col3 = st.columns(3)

            def safe_get_image(drama_name):
                """Returns image URL safely to avoid index errors"""
                try:
                    return df[df['Name'] == drama_name]['img url'].iloc[0]
                except IndexError:
                    return None  # Return None if no image found

            for i, col in enumerate([col1, col2, col3]):
                if i < len(options):
                    with col:
                        img_url = safe_get_image(options[i])
                        if img_url:
                            st.image(img_url)
                        score = st.number_input(f"Pick a score for '{options[i]}'", 1, 10, 1, key=f"score_{i}")
                        ratings.append(score)
                        list_drama_selected.append((df_vector['Name'] == options[i]))

        except Exception as e:
            st.error("An error occurred. Please reselect dramas.")
            st.rerun()  # Restart app to prevent crashes

with col3s:
    if len(options) == 3:
        try:
            df_user = df_vector[df_vector['Name'].isin(options)]
            item_feature_matrix = df_user.loc[:, df_user.columns[1:]]

            item_feature_matrix_with_rating = item_feature_matrix.mul(ratings, axis=0)
            nilai_per_genre = item_feature_matrix_with_rating.sum()
            nilai_total = nilai_per_genre.sum()
            user_feature_vector = nilai_per_genre / nilai_total
            user_feature_vector = user_feature_vector.sort_values(ascending=False).head(10)

            df_unwatched_movies = df_vector[~df_vector['Name'].isin(options)]
            item_feature_matrix_unwatched = df_unwatched_movies.loc[:, df_user.columns[1:]]

            item_feature_unwatched = item_feature_matrix_unwatched.mul(user_feature_vector)
            scoring = item_feature_unwatched.sum(axis=1) * 1000

            df_unwatched_movies_score = df_vector[~df_vector['Name'].isin(options)]
            df_unwatched_movies_score['Movie_Score'] = scoring
            df_unwatched_movies_score_top_5 = df_unwatched_movies_score.sort_values(
                by='Movie_Score', ascending=False).head(5)[['Name', 'Movie_Score']]
            df_unwatched_movies_score_top_5 = df_unwatched_movies_score_top_5.merge(df, on='Name', how='left')

            # Prevent crashes by initializing session state for navigation
            if "score_index" not in st.session_state:
                st.session_state.score_index = 0

            # Navigation functions
            def decrease_score():
                if st.session_state.score_index > 0:
                    st.session_state.score_index -= 1

            def increase_score():
                if st.session_state.score_index < len(df_unwatched_movies_score_top_5) - 1:
                    st.session_state.score_index += 1

            # Loading animation
            with st.spinner("Generating recommendations..."):
                time.sleep(0.5)

            st.markdown('### 5 Recommended K-Dramas:')
            colss1, colss2 = st.columns(2)

            with colss1:
                try:
                    st.image(df_unwatched_movies_score_top_5['img url'].iloc[st.session_state.score_index])
                except IndexError:
                    st.error("Error loading image. Try again.")

            with colss2:
                try:
                    drama_name = df_unwatched_movies_score_top_5['Name'].iloc[st.session_state.score_index]
                    year = df_unwatched_movies_score_top_5['Year'].iloc[st.session_state.score_index]
                    episodes = df_unwatched_movies_score_top_5['Episode'].iloc[st.session_state.score_index]
                    network = df_unwatched_movies_score_top_5['Network'].iloc[st.session_state.score_index]
                    genre = df_unwatched_movies_score_top_5['Genre'].iloc[st.session_state.score_index]
                    cast = df_unwatched_movies_score_top_5['Main Cast'].iloc[st.session_state.score_index]

                    st.markdown(f"**{drama_name}**")
                    st.write(f"**Year:** {year} | **Episodes:** {episodes}")
                    st.write(f"**Network:** {network}")
                    st.write(f"**Genre:** {genre}")
                    st.write(f"**Main Cast:** {cast}")

                    with st.popover("Synopsis"):
                        synopsis = df_unwatched_movies_score_top_5['Sinopsis'].iloc[st.session_state.score_index]
                        st.write(synopsis)
                except IndexError:
                    st.error("Error displaying recommendation. Try again.")

            left, middle, right = st.columns(3)
            with middle:
                st.markdown(f"<h6 style='text-align: center;'> {st.session_state.score_index + 1} / 5 </h6>", unsafe_allow_html=True)

            left.button("Prev", use_container_width=True, on_click=decrease_score)
            right.button("Next", use_container_width=True, on_click=increase_score)

        except Exception as e:
            st.error("Something went wrong. Please try again.")
            st.rerun()  # Restart app to avoid crashes

st.markdown('GitHub: [Source Code](https://github.com/PurwadhikaDev/AlphaGroup_JC_DS_FT_BDG_05_FinalProject/blob/main/README.md?plain=1)')
