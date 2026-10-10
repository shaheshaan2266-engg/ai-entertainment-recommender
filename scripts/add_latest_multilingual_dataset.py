"""
Adds latest 2024-2026 multilingual movies (Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Gujarati)
and comprehensive music (Indian Indie, Marathi, South Indian, English/Global Pop) to the dataset.
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "bollywood"
MOVIES_CSV = DATA_DIR / "bollywood_movies.csv"
SONGS_CSV = DATA_DIR / "bollywood_songs.csv"

EXTRA_MOVIES = [
    # 2024-2026 Bollywood & Hindi
    {
        'title': 'Maharaja',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Thriller, Crime, Drama',
        'director': 'Nithilan Saminathan',
        'actors': 'Vijay Sethupathi, Anurag Kashyap, Mamta Mohandas, Natarajan Subramaniam',
        'overview': 'A humble barber sets out on a relentless quest for justice after a robbery at his home, entangling crooked cops and elusive criminals in a masterclass puzzle.',
        'industry': 'Tamil / Pan-Indian Thriller',
        'imdb_rating': 8.5,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Amaran',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Biography, War, Drama',
        'director': 'Rajkumar Periasamy',
        'actors': 'Sivakarthikeyan, Sai Pallavi, Bhuvan Arora, Rahul Bose',
        'overview': 'The true story of Major Mukund Varadarajan, Ashok Chakra recipient, leading the 44 Rashtriya Rifles in Kashmir while chronicling his eternal love with Indhu Rebecca Varghese.',
        'industry': 'Tamil / Patriotic Biopic',
        'imdb_rating': 8.3,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Premalu',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Romance, Comedy, Slice-of-Life',
        'director': 'Girish A.D.',
        'actors': 'Naslen K. Gafoor, Mamitha Baiju, Shyam Mohan, Sangeeth Prathap',
        'overview': 'Sachin travels to Hyderabad with ambitions to move to the UK, but unexpectedly falls hopelessly for Reenu, triggering hilarious romantic misadventures.',
        'industry': 'Malayalam Cinema / Youth Rom-Com',
        'imdb_rating': 7.9,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Raayan',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Crime, Thriller, Drama',
        'director': 'Dhanush',
        'actors': 'Dhanush, S. J. Suryah, Sundeep Kishan, Kalidas Jayaram, Dushara Vijayan',
        'overview': 'A mild-mannered cook with a dark past in North Madras turns into an unyielding force of fury to protect his younger siblings against ruthless gang bosses.',
        'industry': 'Tamil Cinema / Action Thriller',
        'imdb_rating': 7.5,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Thangalaan',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Historical, Adventure, Drama',
        'director': 'Pa. Ranjith',
        'actors': 'Vikram, Malavika Mohanan, Parvathy Thiruvothu, Daniel Caltagirone',
        'overview': 'Set in the British Raj colonial era, a fierce tribal leader in the Kolar Gold Fields battles sorceresses and colonial oppressors for land, ancestral rights, and gold.',
        'industry': 'Tamil Cinema / Period Epic',
        'imdb_rating': 7.2,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Saripodhaa Sanivaaram',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Thriller, Drama',
        'director': 'Vivek Athreya',
        'actors': 'Nani, S. J. Suryah, Priyanka Mohan, Abhirami',
        'overview': 'Surya controls his uncontrollable rage all week and unleashes righteous vengeance only on Saturdays, bringing him into clash with a corrupt, sadistic police inspector.',
        'industry': 'Telugu Cinema / High-Octane Action',
        'imdb_rating': 7.4,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Lucky Baskhar',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Crime, Drama, Thriller',
        'director': 'Venky Atluri',
        'actors': 'Dulquer Salmaan, Meenakshi Chaudhary, Ramki, Maanasa Choudhary',
        'overview': 'A middle-class bank cashier struggling under financial debt turns to ingenious financial schemes and money laundering during Bombay\'s stock market frenzy of the early 90s.',
        'industry': 'Telugu / Financial Heist Drama',
        'imdb_rating': 8.0,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Article 370',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Political, Thriller, Drama',
        'director': 'Aditya Suhas Dhabolkar',
        'actors': 'Yami Gautam, Priyamani, Kiran Karmarkar, Arun Govil',
        'overview': 'An intelligence officer and a Prime Minister\'s Office bureaucrat navigate high-stakes counter-terrorism and intense geopolitical operations to revoke Article 370 in Kashmir.',
        'industry': 'Bollywood / Political Thriller',
        'imdb_rating': 8.0,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Shaitaan',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Horror, Thriller, Supernatural',
        'director': 'Vikas Bahl',
        'actors': 'Ajay Devgn, R. Madhavan, Jyotika, Janki Bodiwala',
        'overview': 'A family\'s quiet weekend retreat turns into a psychological nightmare when a mysterious stranger hypnotizes their teenage daughter with sinister black magic.',
        'industry': 'Bollywood / Supernatural Horror',
        'imdb_rating': 6.8,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Vettaiyan',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Crime, Drama',
        'director': 'T. J. Gnanavel',
        'actors': 'Rajinikanth, Amitabh Bachchan, Fahadh Faasil, Rana Daggubati, Manju Warrier',
        'overview': 'A ruthless encounter specialist cop faces moral, ethical, and judicial scrutiny from a retired Supreme Court judge following a controversial police operation.',
        'industry': 'Tamil Cinema / Superstar Action',
        'imdb_rating': 7.3,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'ARM (Ajayante Randam Moshanam)',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Fantasy, Adventure, Period',
        'director': 'Jitin Laal',
        'actors': 'Tovino Thomas, Krithi Shetty, Basil Joseph, Aishwarya Rajesh',
        'overview': 'Spanning three generations across Northern Kerala in 1900, 1950, and 1992, three legendary heroes protect their village\'s heavenly celestial treasure forged from a meteor.',
        'industry': 'Malayalam Cinema / 3D Fantasy Epic',
        'imdb_rating': 7.6,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Kishkindha Kaandam',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Mystery, Thriller, Drama',
        'director': 'Dinjith Ayyathan',
        'actors': 'Asif Ali, Aparna Balamurali, Vijayaraghavan, Jagadish',
        'overview': 'In a monkey-inhabited reserve forest, a newly wed couple investigates the mysterious disappearance of a licensed gun and an aging father\'s hidden dementia memories.',
        'industry': 'Malayalam Cinema / Mystery Thriller',
        'imdb_rating': 8.2,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    # Marathi Cinema
    {
        'title': 'Subhedar',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Historical, War, Drama',
        'director': 'Digpal Lanjekar',
        'actors': 'Chinmay Mandlekar, Ajay Purkar, Mrinal Kulkarni, Digpal Lanjekar',
        'overview': 'The epic recapture of Kondhana fort led by Subhedar Tanaji Malusare, who prioritized the saffron flag and the Maratha kingdom over his son\'s impending wedding.',
        'industry': 'Marathi Cinema / Maratha History',
        'imdb_rating': 8.3,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Jhimma 2',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'genres': 'Comedy, Drama, Slice-of-Life',
        'director': 'Hemant Dhome',
        'actors': 'Suhas Joshi, Nirmiti Sawant, Suchitra Bandekar, Kshitee Jog, Sayali Sanjeev',
        'overview': 'A group of vivacious Marathi women reunite for a scenic trip to the United Kingdom, exploring laughter, unresolved personal sorrows, and empowerment.',
        'industry': 'Marathi Cinema / Feel-Good Drama',
        'imdb_rating': 7.5,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    # Gujarati Cinema
    {
        'title': '3 Ekka',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'genres': 'Comedy, Drama',
        'director': 'Rajesh Sharma',
        'actors': 'Malhar Thakar, Yash Soni, Mitra Gadhvi, Esha Kansara, Kinjal Rajpriya',
        'overview': 'Three broke friends concoct an absurd get-rich-quick gamble by turning a simple residential home into an underground casino, inviting madcap chaos.',
        'industry': 'Gujarati Cinema / Comedy Blockbuster',
        'imdb_rating': 7.7,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Fakt Mahilao Maate',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Comedy, Fantasy, Drama',
        'director': 'Jay Bodas',
        'actors': 'Yash Soni, Amitabh Bachchan, Deeksha Joshi, Tarjanee Bhadla',
        'overview': 'A frustrated young man who struggles to understand women is magically granted the supernatural superpower to hear women\'s inner thoughts.',
        'industry': 'Gujarati Cinema / Fantasy Comedy',
        'imdb_rating': 7.4,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    }
]

EXTRA_SONGS = [
    # Indian Indie & Acoustic
    {
        'title': 'Alag Aasmaan',
        'movie_title': 'Indie Singles',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'Anuv Jain',
        'composer': 'Anuv Jain',
        'lyricist': 'Anuv Jain',
        'mood': 'Indie & Acoustic, Romantic Melancholy',
        'thumbnail_url': 'https://i.ytimg.com/vi/RLzC55ai0eo/hqdefault.jpg',
        'lyrics_snippet': 'Nayi nayi si baatein hai aisi... Intimate long-distance acoustic love ballad celebrating tender moments under different skies.'
    },
    {
        'title': 'Mishri',
        'movie_title': 'Indie Singles',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'Anuv Jain',
        'composer': 'Anuv Jain',
        'lyricist': 'Anuv Jain',
        'mood': 'Indie & Acoustic, Soulful Melodies',
        'thumbnail_url': 'https://i.ytimg.com/vi/RLzC55ai0eo/hqdefault.jpg',
        'lyrics_snippet': 'Yeh meethi meethi si baatein... Warm acoustic poetry with soft fingerstyle guitar and gentle melodies.'
    },
    {
        'title': 'Maan Meri Jaan',
        'movie_title': 'Champagne Talk',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'singer': 'King',
        'composer': 'King, Saurabh Lokhande',
        'lyricist': 'King',
        'mood': 'Indian Indie, Hip Hop & Pop, Romantic Anthem',
        'thumbnail_url': 'https://i.ytimg.com/vi/VuG7FT9evQ8/hqdefault.jpg',
        'lyrics_snippet': 'Maan meri jaan main tujhe jaane na doonga... Global viral Indian indie-pop romantic anthem blending modern beats with emotional Hindi vocals.'
    },
    {
        'title': 'Tu Aake Dekhle',
        'movie_title': 'The Carnival',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'King',
        'composer': 'King, Shahrule',
        'lyricist': 'King',
        'mood': 'Indian Indie, Hip Hop, Pop',
        'thumbnail_url': 'https://i.ytimg.com/vi/A66TYF3EQcQ/hqdefault.jpg',
        'lyrics_snippet': 'Tu aake dekhle ho maine raatein kitni do sari... Breakout chart-topping melodic rap song that captured millions.'
    },
    {
        'title': '3:59 AM',
        'movie_title': 'Punya Paap',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'DIVINE',
        'composer': 'Stunnah Beatz',
        'lyricist': 'DIVINE',
        'mood': 'Hip Hop & Rap, Desi Hip Hop, Hard Hitting',
        'thumbnail_url': 'https://i.ytimg.com/vi/8nTRv0lQp3E/hqdefault.jpg',
        'lyrics_snippet': 'Chhote gaane me jab aate tab sabki fat jaati... The pinnacle of Indian underground hip hop lyricism and Gully rap intensity.'
    },
    {
        'title': 'Kho Gaye Hum Kahan',
        'movie_title': 'Baar Baar Dekho',
        'release_year': 2016,
        'era': '2010s Blockbusters',
        'singer': 'Jasleen Royal, Prateek Kuhad',
        'composer': 'Jasleen Royal',
        'lyricist': 'Anvita Dutt',
        'mood': 'Indie & Acoustic, Soulful, Romantic',
        'thumbnail_url': 'https://i.ytimg.com/vi/Il7nvbU-7cI/hqdefault.jpg',
        'lyrics_snippet': 'Kho gaye hum kahan, rangon sa yeh jahan... Dreamy acoustic indie pop collaboration about being lost in love.'
    },
    # South Indian Chartbusters
    {
        'title': 'Illuminati',
        'movie_title': 'Aavesham',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'singer': 'Dabzee',
        'composer': 'Sushin Shyam',
        'lyricist': 'Vinayak Sasikumar',
        'mood': 'Dance & Electronic, Hip Hop, Malayalam Viral Hit',
        'thumbnail_url': 'https://i.ytimg.com/vi/tOM-nWPcR4U/hqdefault.jpg',
        'lyrics_snippet': 'Vibe machane Illuminati... High-voltage Malayalam party explosion dominating Instagram reels across the globe.'
    },
    {
        'title': 'Katchi Sera',
        'movie_title': 'Independent Singles',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'singer': 'Sai Abhyankkar',
        'composer': 'Sai Abhyankkar',
        'lyricist': 'Adesh Krishna',
        'mood': 'Tamil Pop, Indie Dance, Romantic Groove',
        'thumbnail_url': 'https://i.ytimg.com/vi/fB8UTheTR7s/hqdefault.jpg',
        'lyrics_snippet': 'Katchi sera katchi sera vanthuttaen... Mega viral Tamil indie pop sensation with addictive grooves and soul-stirring rhythm.'
    },
    {
        'title': 'Rowdy Baby',
        'movie_title': 'Maari 2',
        'release_year': 2018,
        'era': '2010s Blockbusters',
        'singer': 'Dhanush, Dhee',
        'composer': 'Yuvan Shankar Raja',
        'lyricist': 'Dhanush',
        'mood': 'Dance, Fast Folk, Tamil Blockbuster',
        'thumbnail_url': 'https://i.ytimg.com/vi/x6Q7c9RyMzk/hqdefault.jpg',
        'lyrics_snippet': 'Hey rowdy baby, en rowdy baby... One of the most viewed Indian music videos of all time with over 1.5 billion views.'
    },
    # Marathi Music Hits
    {
        'title': 'Dolby Walya',
        'movie_title': 'Jallosh',
        'release_year': 2016,
        'era': '2010s Blockbusters',
        'singer': 'Nagesh Morwekar',
        'composer': 'Pravin Kunwar',
        'lyricist': 'Sameer Samant',
        'mood': 'Dance, Marathi Dhol Beats, High Energy',
        'thumbnail_url': 'https://i.ytimg.com/vi/sFA9H9vNlHk/hqdefault.jpg',
        'lyrics_snippet': 'Dolby walya bolav mazya dj la... Electrifying Marathi celebration song played at every festival and Ganesh Utsav.'
    },
    {
        'title': 'Kombdi Palali',
        'movie_title': 'Jatra',
        'release_year': 2006,
        'era': '2000s Hits',
        'singer': 'Anand Shinde, Vaishali Samant',
        'composer': 'Ajay-Atul',
        'lyricist': 'Jitendra Joshi',
        'mood': 'Folk Dance, Marathi Classic, Fast Groovy',
        'thumbnail_url': 'https://i.ytimg.com/vi/RLzC55ai0eo/hqdefault.jpg',
        'lyrics_snippet': 'Kombdi palali tangdi dharun langdi ghalali... Legendary Marathi dance anthem composed by Ajay-Atul that later inspired Chikni Chameli.'
    },
    # Global & English Pop
    {
        'title': 'Anti-Hero',
        'movie_title': 'Midnights',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'singer': 'Taylor Swift',
        'composer': 'Taylor Swift, Jack Antonoff',
        'lyricist': 'Taylor Swift, Jack Antonoff',
        'mood': 'English Pop, Synth-Pop, Introspective',
        'thumbnail_url': 'https://i.ytimg.com/vi/b1kbLwvqugk/hqdefault.jpg',
        'lyrics_snippet': "It's me, hi, I'm the problem, it's me... Billboard #1 smash hit examining self-loathing, fame, and vulnerability."
    },
    {
        'title': 'Starboy',
        'movie_title': 'Starboy Album',
        'release_year': 2016,
        'era': '2010s Blockbusters',
        'singer': 'The Weeknd, Daft Punk',
        'composer': 'The Weeknd, Daft Punk, Doc McKinney',
        'lyricist': 'The Weeknd',
        'mood': 'English Pop, R&B, Electronic, Electro-Hop',
        'thumbnail_url': 'https://i.ytimg.com/vi/34Na4j8AVgA/hqdefault.jpg',
        'lyrics_snippet': "Look what you've done, I'm a motherf*ckin' starboy... High-energy synth and bass collaboration with Daft Punk."
    },
    {
        'title': 'Save Your Tears',
        'movie_title': 'After Hours',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'The Weeknd',
        'composer': 'The Weeknd, Max Martin, Oscar Holter',
        'lyricist': 'The Weeknd',
        'mood': 'English Pop, Synthwave, Melodic Romance',
        'thumbnail_url': 'https://i.ytimg.com/vi/XXYlFuWEuKI/hqdefault.jpg',
        'lyrics_snippet': "Save your tears for another day... 80s nostalgia synth-pop masterpiece beloved worldwide."
    },
    {
        'title': 'Die With A Smile',
        'movie_title': 'Global Singles',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'singer': 'Lady Gaga, Bruno Mars',
        'composer': 'Bruno Mars, Lady Gaga, Andrew Watt',
        'lyricist': 'Bruno Mars, Lady Gaga',
        'mood': 'English Pop, Soul Ballad, Romantic Duet',
        'thumbnail_url': 'https://i.ytimg.com/vi/kPa7bsKwL-8/hqdefault.jpg',
        'lyrics_snippet': "If the world was ending, I'd wanna be next to you... The biggest global vocal duet of 2024."
    },
    {
        'title': 'As It Was',
        'movie_title': "Harry's House",
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'singer': 'Harry Styles',
        'composer': 'Harry Styles, Kid Harpoon, Tyler Johnson',
        'lyricist': 'Harry Styles',
        'mood': 'English Pop, Synth-Pop, Indie Pop, Nostalgia',
        'thumbnail_url': 'https://i.ytimg.com/vi/H5v3kku4y6Q/hqdefault.jpg',
        'lyrics_snippet': "You know it's not the same as it was... Record-breaking global indie-pop anthem."
    }
]

def main():
    df_movies = pd.read_csv(MOVIES_CSV)
    df_songs = pd.read_csv(SONGS_CSV)

    existing_movie_titles = set(df_movies['title'].str.lower().str.strip())
    existing_song_titles = set(df_songs['title'].str.lower().str.strip())

    new_m = [m for m in EXTRA_MOVIES if m['title'].lower().strip() not in existing_movie_titles]
    new_s = [s for s in EXTRA_SONGS if s['title'].lower().strip() not in existing_song_titles]

    if new_m:
        df_movies = pd.concat([df_movies, pd.DataFrame(new_m)], ignore_index=True)
        df_movies.to_csv(MOVIES_CSV, index=False)
        print(f"Added {len(new_m)} new movies. Total movies now: {len(df_movies)}")
    else:
        print(f"All movies already present. Total: {len(df_movies)}")

    if new_s:
        df_songs = pd.concat([df_songs, pd.DataFrame(new_s)], ignore_index=True)
        df_songs.to_csv(SONGS_CSV, index=False)
        print(f"Added {len(new_s)} new songs. Total songs now: {len(df_songs)}")
    else:
        print(f"All songs already present. Total: {len(df_songs)}")

if __name__ == "__main__":
    main()
