"""
Comprehensive dataset enrichment script for Velora AI Recommender.
Enriches movies and songs datasets with:
- Recent Latest Indian Movies (2024-2026)
- Bollywood & Hindi cinema
- South Indian Cinema: Tamil, Telugu, Kannada, Malayalam
- Marathi Cinema
- Gujarati Cinema
- Indian Independent & Indie Music
- Marathi Music
- South Indian Soundtracks
- English & Global Pop Music
- Multilingual song & movie connections
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "bollywood"
MOVIES_CSV = DATA_DIR / "bollywood_movies.csv"
SONGS_CSV = DATA_DIR / "bollywood_songs.csv"

# ============================================================================
# 1. EXPANDED MOVIES CATALOG
# ============================================================================
EXPANDED_MOVIES = [
    # --- 2024-2026 Recent Bollywood & Hindi Cinema ---
    {
        'title': 'Chhaava',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Historical, Biography, Epic, Drama',
        'director': 'Laxman Utekar',
        'actors': 'Vicky Kaushal, Rashmika Mandanna, Akshaye Khanna, Ashutosh Rana',
        'overview': 'The epic historical chronicle of Chhatrapati Sambhaji Maharaj, the valiant son of Shivaji Maharaj, leading the Maratha Empire against the Mughal emperor Aurangzeb.',
        'industry': 'Bollywood / Historical Epic',
        'imdb_rating': 8.4,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Stree 2',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Comedy, Horror, Mystery, Fantasy',
        'director': 'Amar Kaushik',
        'actors': 'Shraddha Kapoor, Rajkummar Rao, Pankaj Tripathi, Abhishek Banerjee, Aparshakti Khurana',
        'overview': 'After the events of Stree, the town of Chanderi faces a new headless terror named Sarkata. Vicky and his loyal friends team up with Stree to save the women of the town.',
        'industry': 'Bollywood / Horror-Comedy Universe',
        'imdb_rating': 7.6,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Bhool Bhulaiyaa 3',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Comedy, Horror, Mystery, Drama',
        'director': 'Anees Bazmee',
        'actors': 'Kartik Aaryan, Vidya Balan, Madhuri Dixit, Triptii Dimri, Rajpal Yadav',
        'overview': 'Rooh Baba returns to face the vengeful spirit of Manjulika inside the ancient palace of Raktaghat, leading to terrifying supernatural confrontations and hilarious chaos.',
        'industry': 'Bollywood',
        'imdb_rating': 7.0,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Singham Again',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Crime, Drama, Thriller',
        'director': 'Rohit Shetty',
        'actors': 'Ajay Devgn, Kareena Kapoor, Ranveer Singh, Akshay Kumar, Deepika Padukone, Tiger Shroff, Arjun Kapoor',
        'overview': 'Bajirao Singham traverses across borders to rescue his kidnapped wife in a modern high-octane narrative drawing parallels to the epic Ramayana alongside his Cop Universe allies.',
        'industry': 'Bollywood / Cop Universe',
        'imdb_rating': 6.8,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Fighter',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Thriller, War, Drama',
        'director': 'Siddharth Anand',
        'actors': 'Hrithik Roshan, Deepika Padukone, Anil Kapoor, Karan Singh Grover, Akshay Oberoi',
        'overview': 'Top Indian Air Force aviators come together to form the Air Dragons unit, facing mortal combat and soaring aerial dogfights in response to terrorist attacks.',
        'industry': 'Bollywood',
        'imdb_rating': 7.3,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': '12th Fail',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'genres': 'Biography, Drama, Inspirational',
        'director': 'Vidhu Vinod Chopra',
        'actors': 'Vikrant Massey, Medha Shankr, Anant V Joshi, Priyanshu Chatterjee',
        'overview': 'The real-life story of IPS officer Manoj Kumar Sharma who restarts his academic journey from absolute poverty to conquer the toughest civil services examination in India.',
        'industry': 'Bollywood',
        'imdb_rating': 8.9,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Kill',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Thriller, Crime',
        'director': 'Nikhil Nagesh Bhat',
        'actors': 'Lakshya, Raghav Juyal, Tanya Maniktala, Ashish Vidyarthi',
        'overview': 'When a train to New Delhi is invaded by a gang of bloodthirsty bandits, army commando Amrit wages an unrelenting, visceral close-quarters battle to protect the passengers and his beloved.',
        'industry': 'Bollywood / Action Landmark',
        'imdb_rating': 7.6,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Munjya',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Comedy, Horror, Folklore, Fantasy',
        'director': 'Aditya Sarpotdar',
        'actors': 'Abhay Verma, Sharvari Wagh, Mona Singh, S. Sathyaraj',
        'overview': 'A young man visits his ancestral village in Konkan and unearths an obsessive folklore spirit called Munjya who wants to compel him into marriage.',
        'industry': 'Bollywood / Konkan Folklore',
        'imdb_rating': 7.1,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Chandu Champion',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Biography, Drama, Sport',
        'director': 'Kabir Khan',
        'actors': 'Kartik Aaryan, Vijay Raaz, Bhuvan Arora, Yashpal Sharma',
        'overview': 'The extraordinary true story of Murlikant Petkar, India’s first Paralympic gold medalist, who survived nine bullets during the 1965 war and defied all odds to become a champion.',
        'industry': 'Bollywood',
        'imdb_rating': 8.1,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },

    # --- Telugu Cinema (Tollywood / Pan-India) ---
    {
        'title': 'Pushpa 2: The Rule',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Crime, Drama, Thriller',
        'director': 'Sukumar',
        'actors': 'Allu Arjun, Rashmika Mandanna, Fahadh Faasil, Sunil, Jagapathi Babu',
        'overview': 'Pushpa Raj expands his red sandalwood empire into an international syndicate while waging total war against Bhanwar Singh Shekhawat in an explosive showdown.',
        'industry': 'Tollywood / Pan-India',
        'imdb_rating': 8.1,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Kalki 2898 AD',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Sci-Fi, Mythological, Adventure',
        'director': 'Nag Ashwin',
        'actors': 'Prabhas, Amitabh Bachchan, Kamal Haasan, Deepika Padukone, Disha Patani',
        'overview': 'In a dystopian post-apocalyptic future in Kasi, the immortal warrior Ashwatthama awakens to protect SUM-80 and the unborn child foretold to herald the tenth avatar of Vishnu.',
        'industry': 'Tollywood / Pan-India Sci-Fi',
        'imdb_rating': 7.6,
        'poster_url': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Kalki_2898_AD.jpg/220px-Kalki_2898_AD.jpg'
    },
    {
        'title': 'Devara: Part 1',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Drama, Thriller, Epic',
        'director': 'Koratala Siva',
        'actors': 'N.T. Rama Rao Jr., Saif Ali Khan, Janhvi Kapoor, Prakash Raj, Srikanth',
        'overview': 'A fearless coastal chieftain Devara stands up against weapon smuggling across treacherous waters, and his timid son Vara must inherit his legacy to protect their clan.',
        'industry': 'Tollywood / Pan-India',
        'imdb_rating': 7.2,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Hanu-Man',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Adventure, Fantasy, Superhero',
        'director': 'Prasanth Varma',
        'actors': 'Teja Sajja, Amritha Aiyer, Varalaxmi Sarathkumar, Vinay Rai',
        'overview': 'An underdog villager in Saurashtra accidentally discovers a sacred jewel possessing the infinite powers of Lord Hanuman and must defend his land against a ruthless supervillain.',
        'industry': 'Tollywood / Prasanth Varma Cinematic Universe',
        'imdb_rating': 7.9,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Salaar: Part 1 – Ceasefire',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Crime, Drama, Thriller',
        'director': 'Prashanth Neel',
        'actors': 'Prabhas, Prithviraj Sukumaran, Shruti Haasan, Jagapathi Babu, Bobby Simha',
        'overview': 'The brutal city-state of Khansaar descends into dynastic warfare, prompting prince Varadha to call upon his exiled childhood friend Deva, the unstoppable one-man army.',
        'industry': 'Tollywood / Pan-India',
        'imdb_rating': 7.5,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Sita Ramam',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Drama, Mystery, Romance, Period',
        'director': 'Hanu Raghavapudi',
        'actors': 'Dulquer Salmaan, Mrunal Thakur, Rashmika Mandanna, Sumanth',
        'overview': 'An orphaned Indian army lieutenant stationed in Kashmir receives anonymous letters from a woman named Sita Mahalakshmi, sparking an unforgettable love that defies national borders.',
        'industry': 'Tollywood / Pan-India Romance',
        'imdb_rating': 8.5,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },

    # --- Tamil Cinema (Kollywood) ---
    {
        'title': 'Leo',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Crime, Thriller, Drama',
        'director': 'Lokesh Kanagaraj',
        'actors': 'Thalapathy Vijay, Sanjay Dutt, Arjun Sarja, Trisha Krishnan, Gautham Vasudev Menon',
        'overview': 'Parthiban, a quiet cafe owner in Himachal Pradesh, is pursued by an international cartel who suspect him of being the legendary gangster Leo Das.',
        'industry': 'Kollywood / LCU',
        'imdb_rating': 7.8,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Vikram',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Crime, Thriller, Mystery',
        'director': 'Lokesh Kanagaraj',
        'actors': 'Kamal Haasan, Vijay Sethupathi, Fahadh Faasil, Suriya, Narain',
        'overview': 'A black-ops squad leader investigates a serial masked assassination ring, uncovering a vast drug empire led by Santhanam and the shadowy cartel commander Rolex.',
        'industry': 'Kollywood / LCU',
        'imdb_rating': 8.3,
        'poster_url': 'https://upload.wikimedia.org/wikipedia/en/thumb/9/93/Vikram_2022_poster.jpg/220px-Vikram_2022_poster.jpg'
    },
    {
        'title': 'Ponniyin Selvan: Part 1',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Adventure, Drama, Historical, Epic',
        'director': 'Mani Ratnam',
        'actors': 'Vikram, Aishwarya Rai Bachchan, Jayam Ravi, Karthi, Trisha Krishnan',
        'overview': 'Vandiyathevan embarks on a dangerous diplomatic mission across southern India to thwart a conspiracy against the Chola empire and crown prince Aditha Karikalan.',
        'industry': 'Kollywood / Epic History',
        'imdb_rating': 7.6,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'The Greatest of All Time (GOAT)',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Sci-Fi, Thriller',
        'director': 'Venkat Prabhu',
        'actors': 'Thalapathy Vijay, Prashanth, Prabhu Deva, Mohan, Jayaram, Sneha',
        'overview': 'A veteran field operative of the Special Anti-Terrorist Squad confronts the deadliest threat to his nation and family: a lethal younger clone of himself.',
        'industry': 'Kollywood',
        'imdb_rating': 6.9,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Jai Bhim',
        'release_year': 2021,
        'era': 'Latest (2020-2026)',
        'genres': 'Crime, Drama, Legal, Mystery',
        'director': 'T.J. Gnanavel',
        'actors': 'Suriya, Lijomol Jose, Manikandan, Prakash Raj, Rajisha Vijayan',
        'overview': 'When a tribal man is falsely detained and goes missing from police custody, a brave human rights lawyer files a Habeas Corpus petition to fight for justice against systemic tyranny.',
        'industry': 'Kollywood / Social Landmark',
        'imdb_rating': 8.8,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },

    # --- Malayalam Cinema (Mollywood) ---
    {
        'title': 'Manjummel Boys',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Adventure, Drama, Survival, Thriller',
        'director': 'Chidambaram',
        'actors': 'Soubin Shahir, Sreenath Bhasi, Balu Varghese, Ganapathi',
        'overview': 'A close-knit group of friends from Kochi travel to Kodaikanal, where one of them falls into the dreaded Guna Caves, triggering an extraordinary real-life rescue against impossible odds.',
        'industry': 'Mollywood / Global Sensation',
        'imdb_rating': 8.5,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Aavesham',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Comedy, Crime',
        'director': 'Jithu Madhavan',
        'actors': 'Fahadh Faasil, Hipzster, Mithun Jai Shankar, Roshan Shanavas',
        'overview': 'Three engineering college students in Bengaluru who are bullied by seniors befriend an eccentric local gangster named Ranga Anna, sparking riotous comedic and violent adventures.',
        'industry': 'Mollywood',
        'imdb_rating': 7.9,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Bramayugam',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Horror, Mystery, Thriller, Period',
        'director': 'Rahul Sadasivan',
        'actors': 'Mammootty, Arjun Ashokan, Sidharth Bharathan',
        'overview': 'In 17th-century Malabar, a wandering folk singer seeking refuge arrives at a decaying feudal mansion owned by an enigmatic sorcerer, only to find himself trapped in a nightmare.',
        'industry': 'Mollywood / Black-and-White Horror',
        'imdb_rating': 7.9,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'The Goat Life (Aadujeevitham)',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Adventure, Biography, Drama, Survival',
        'director': 'Blessy',
        'actors': 'Prithviraj Sukumaran, Jimmy Jean-Louis, KR Gokul, Amala Paul',
        'overview': 'The harrowing true survival saga of Najeeb, an Indian migrant worker who travels to Saudi Arabia only to be enslaved as a goat herder in the merciless Arabian desert.',
        'industry': 'Mollywood / International Drama',
        'imdb_rating': 8.2,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },

    # --- Kannada Cinema (Sandalwood) ---
    {
        'title': 'Kantara',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Drama, Folklore, Thriller, Mystery',
        'director': 'Rishab Shetty',
        'actors': 'Rishab Shetty, Sapthami Gowda, Kishore Kumar G., Achyuth Kumar',
        'overview': 'In the coastal forests of Karnataka, a rebellious Kambala champion clashes with a forest officer, awakening the sacred ancestral demi-god Panjurli Daiva.',
        'industry': 'Sandalwood / Pan-India Folklore',
        'imdb_rating': 8.2,
        'poster_url': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Kantara_poster.jpeg/220px-Kantara_poster.jpeg'
    },
    {
        'title': '777 Charlie',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Adventure, Comedy, Drama',
        'director': 'Kiranraj K',
        'actors': 'Rakshit Shetty, Sangeetha Sringeri, Raj B. Shetty, Danish Sait',
        'overview': 'Dharma, a solitary and misanthropic factory worker, finds his life transformed when a shelter Labrador named Charlie enters his world and takes him on a journey to Kashmir.',
        'industry': 'Sandalwood',
        'imdb_rating': 8.8,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },

    # --- Marathi Cinema ---
    {
        'title': 'Sairat',
        'release_year': 2016,
        'era': '2010s Blockbusters',
        'genres': 'Drama, Romance, Musical',
        'director': 'Nagraj Manjule',
        'actors': 'Rinku Rajguru, Akash Thosar, Tanaji Galgunde, Arbaz Shaikh',
        'overview': 'Two college students from differing castes fall passionately in love in rural Maharashtra and brave community prejudice in this historic box-office phenomenon.',
        'industry': 'Marathi Cinema / Cultural Landmark',
        'imdb_rating': 8.3,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Natsamrat',
        'release_year': 2016,
        'era': '2010s Blockbusters',
        'genres': 'Drama, Family',
        'director': 'Mahesh Manjrekar',
        'actors': 'Nana Patekar, Medha Manjrekar, Vikram Gokhale, Sunil Barve',
        'overview': 'A revered Shakespearean theatre actor retires from stage life and divides his wealth among his children, only to suffer heartbreak and abandonment in his twilight years.',
        'industry': 'Marathi Cinema',
        'imdb_rating': 8.9,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Baipan Bhaari Deva',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'genres': 'Comedy, Drama, Family',
        'director': 'Kedar Shinde',
        'actors': 'Rohini Hattangadi, Vandana Gupte, Sukanya Kulkarni, Shilpa Navalkar, Suchitra Bandekar, Deepa Parab',
        'overview': 'Six estranged sisters with unique personal struggles reunite to participate in a traditional Mangalagaur competition, rediscovering their unbreakable sisterhood.',
        'industry': 'Marathi Cinema / Blockbuster',
        'imdb_rating': 7.6,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Pawankhind',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Historical, War, Epic',
        'director': 'Digpal Lanjekar',
        'actors': 'Chinmay Mandlekar, Ajay Purkar, Sameer Dharmadhikari, Ankit Mohan',
        'overview': 'The historic Battle of Pavan Khind, where Maratha warrior Baji Prabhu Deshpande and his 300 soldiers held off 10,000 Adilshahi soldiers to ensure Shivaji Maharaj reached safety.',
        'industry': 'Marathi Cinema / Maratha History',
        'imdb_rating': 8.4,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Katyar Kaljat Ghusali',
        'release_year': 2015,
        'era': '2010s Blockbusters',
        'genres': 'Drama, Musical, Classical',
        'director': 'Subodh Bhave',
        'actors': 'Sachin Pilgaonkar, Shankar Mahadevan, Subodh Bhave, Amruta Khanvilkar',
        'overview': 'A musical duel spanning decades between two legendary classical vocal maestros for the royal patronage of the Vishrampur court, set to immortal ragas.',
        'industry': 'Marathi Cinema / Musical Masterpiece',
        'imdb_rating': 8.6,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Ved',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'genres': 'Drama, Romance, Sport',
        'director': 'Riteish Deshmukh',
        'actors': 'Riteish Deshmukh, Genelia Deshmukh, Jiya Shankar, Ashok Saraf',
        'overview': 'Satya, a former cricketer broken by unfulfilled love, enters a marriage of convenience with Shravani, unaware that she has quietly loved him since childhood.',
        'industry': 'Marathi Cinema',
        'imdb_rating': 7.3,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },

    # --- Gujarati Cinema ---
    {
        'title': 'Chhello Show (Last Film Show)',
        'release_year': 2021,
        'era': 'Latest (2020-2026)',
        'genres': 'Drama, Coming-of-Age',
        'director': 'Pan Nalin',
        'actors': 'Bhavin Rabari, Bhavesh Shrimali, Richa Meena, Dipen Raval',
        'overview': 'A 9-year-old boy in rural Chalala falls unconditionally in love with cinema, bribing a projectionist with his mother’s home-cooked meals to watch movies from the booth.',
        'industry': 'Gujarati Cinema / Oscar Nominee',
        'imdb_rating': 7.6,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    },
    {
        'title': 'Hellaro',
        'release_year': 2019,
        'era': '2010s Blockbusters',
        'genres': 'Drama, Musical, Period',
        'director': 'Abhishek Shah',
        'actors': 'Shraddha Dangar, Jayesh More, Niilam Paanchal, Kaushambi Bhatt',
        'overview': 'In 1975 in drought-parched Kutch, a group of repressed women discover joy and liberation through the hypnotic beats of a wandering dhol player, defying orthodox patriarchal laws.',
        'industry': 'Gujarati Cinema / National Award Best Feature',
        'imdb_rating': 8.5,
        'poster_url': 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&q=80'
    },
    {
        'title': 'Chaal Jeevi Laiye!',
        'release_year': 2019,
        'era': '2010s Blockbusters',
        'genres': 'Comedy, Drama, Adventure, Family',
        'director': 'Vipul Mehta',
        'actors': 'Siddharth Randeria, Yash Soni, Aarohi Patel, Jagesh Mukati',
        'overview': 'A workaholic son takes his terminally ill father on an impromptu road trip across Uttarakhand, leading to life-altering encounters with a free-spirited backpacker.',
        'industry': 'Gujarati Cinema / All-Time Blockbuster',
        'imdb_rating': 8.2,
        'poster_url': 'https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&q=80'
    },
    {
        'title': 'Kasoombo',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'genres': 'Action, Historical, War, Epic',
        'director': 'Vijaygiri Bava',
        'actors': 'Raunaq Kamdar, Dharmendra Gohil, Darshan Pandya, Monal Gajjar',
        'overview': 'The valorous resistance of Dadu Barot and 51 Barot warriors who laid down their lives to defend the sacred temples of Palitana against Allauddin Khilji’s commander.',
        'industry': 'Gujarati Cinema / Historical Epic',
        'imdb_rating': 8.6,
        'poster_url': 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&q=80'
    }
]

# ============================================================================
# 2. EXPANDED MUSIC & SONGS CATALOG
# ============================================================================
EXPANDED_SONGS = [
    # --- Indian Independent & Indie Music ---
    {
        'title': 'Husn',
        'movie_title': 'Indie Singles',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'singer': 'Anuv Jain',
        'composer': 'Anuv Jain',
        'lyricist': 'Anuv Jain',
        'mood': 'Indie & Acoustic, Romantic & Soulful',
        'thumbnail_url': 'https://i.ytimg.com/vi/gjhmsGz3JvI/hqdefault.jpg',
        'lyrics_snippet': 'Dekho dekho kaise baatein yahan banti hain... Poignant viral independent acoustic ballad about unspoken longing and one-sided romance.'
    },
    {
        'title': 'Baarishein',
        'movie_title': 'Indie Singles',
        'release_year': 2018,
        'era': '2010s Blockbusters',
        'singer': 'Anuv Jain',
        'composer': 'Anuv Jain',
        'lyricist': 'Anuv Jain',
        'mood': 'Indie & Acoustic, Melancholy',
        'thumbnail_url': 'https://i.ytimg.com/vi/RLzC55ai0eo/hqdefault.jpg',
        'lyrics_snippet': 'Haule se dheeme se mujhko gale lagaana... Iconic acoustic heartbreak anthem with intimate guitar strumming by Anuv Jain.'
    },
    {
        'title': 'cold/mess',
        'movie_title': 'cold/mess EP',
        'release_year': 2018,
        'era': '2010s Blockbusters',
        'singer': 'Prateek Kuhad',
        'composer': 'Prateek Kuhad',
        'lyricist': 'Prateek Kuhad',
        'mood': 'Indie & Acoustic, English Pop, Soulful',
        'thumbnail_url': 'https://i.ytimg.com/vi/Il7nvbU-7cI/hqdefault.jpg',
        'lyrics_snippet': 'When I feel cold I keep you close... Internationally acclaimed indie-pop ballad celebrating vulnerability and turbulent love.'
    },
    {
        'title': 'Kasoor',
        'movie_title': 'Indie Singles',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'Prateek Kuhad',
        'composer': 'Prateek Kuhad',
        'lyricist': 'Prateek Kuhad',
        'mood': 'Indie & Acoustic, Heartwarming',
        'thumbnail_url': 'https://i.ytimg.com/vi/BmUe3-78n9E/hqdefault.jpg',
        'lyrics_snippet': 'Kya kasoor hai mera jo akele khada hoon... Tender indie folk exploration of nostalgia, innocence, and young romance.'
    },
    {
        'title': 'Choo Lo',
        'movie_title': 'Aalas Ka Pedh',
        'release_year': 2015,
        'era': '2010s Blockbusters',
        'singer': 'The Local Train, Raman Negi',
        'composer': 'The Local Train',
        'lyricist': 'Raman Negi',
        'mood': 'Rock & Acoustic, Indie Pop, Nostalgia',
        'thumbnail_url': 'https://i.ytimg.com/vi/sFA9H9vNlHk/hqdefault.jpg',
        'lyrics_snippet': 'Khada hoon aaj bhi wahin ke dil phir beqarar hai... The supreme cult anthem of Indian independent Hindi rock music.'
    },
    {
        'title': 'Aaoge Tum Kabhi',
        'movie_title': 'Aalas Ka Pedh',
        'release_year': 2015,
        'era': '2010s Blockbusters',
        'singer': 'The Local Train',
        'composer': 'The Local Train',
        'lyricist': 'Raman Negi',
        'mood': 'Rock & Acoustic, Hopeful Romance',
        'thumbnail_url': 'https://i.ytimg.com/vi/eH3GzZxPv2Q/hqdefault.jpg',
        'lyrics_snippet': 'Aaoge tum kabhi meri jaan keh rahi... Soaring indie rock ballad filled with electric guitars and longing.'
    },
    {
        'title': 'Udd Gaye',
        'movie_title': 'Bacardi House Party Sessions',
        'release_year': 2017,
        'era': '2010s Blockbusters',
        'singer': 'Ritviz',
        'composer': 'Ritviz',
        'lyricist': 'Ritviz',
        'mood': 'Dance & Electronic, Indian Indie, High Energy',
        'thumbnail_url': 'https://i.ytimg.com/vi/v2-9rIL_f4w/hqdefault.jpg',
        'lyrics_snippet': 'Khwabon ke darmiyan udaan yeh chal rahi... Groundbreaking electronic-pop fusion featuring classical Hindustani flourishes and heavy bass.'
    },
    {
        'title': 'Sage',
        'movie_title': 'VED',
        'release_year': 2019,
        'era': '2010s Blockbusters',
        'singer': 'Ritviz',
        'composer': 'Ritviz',
        'lyricist': 'Ritviz',
        'mood': 'Dance & Electronic, Indian Indie',
        'thumbnail_url': 'https://i.ytimg.com/vi/K4DyBUG242c/hqdefault.jpg',
        'lyrics_snippet': 'Ek dafa jo tu muskuraaye... Infectious indie-electronic melody with classical vocal runs and energetic synths.'
    },
    {
        'title': 'Mirchi',
        'movie_title': 'Punya Paap',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'DIVINE, MC Altaf, Phenom, Stylo G',
        'composer': 'Phenom, DIVINE',
        'lyricist': 'DIVINE, MC Altaf',
        'mood': 'Hip Hop & Rap, High Energy',
        'thumbnail_url': 'https://i.ytimg.com/vi/jZ9a3tE30t4/hqdefault.jpg',
        'lyrics_snippet': 'Teekhi teekhi tu mirchi hai... Street rap blockbuster that put gully hip hop on the international map.'
    },

    # --- Marathi Hit Melodies ---
    {
        'title': 'Zingaat',
        'movie_title': 'Sairat',
        'release_year': 2016,
        'era': '2010s Blockbusters',
        'singer': 'Ajay-Atul',
        'composer': 'Ajay-Atul',
        'lyricist': 'Ajay-Atul',
        'mood': 'High Energy & Dance, Folk Fusion',
        'thumbnail_url': 'https://i.ytimg.com/vi/8ZhnGkX_oUQ/hqdefault.jpg',
        'lyrics_snippet': 'Urat hoti dhak dhak jeev maza kora... Universal pan-Indian energetic dance hurricane composed by the legendary Ajay-Atul.'
    },
    {
        'title': 'Yad Lagla',
        'movie_title': 'Sairat',
        'release_year': 2016,
        'era': '2010s Blockbusters',
        'singer': 'Ajay Gogavale',
        'composer': 'Ajay-Atul',
        'lyricist': 'Ajay-Atul',
        'mood': 'Romantic & Soulful, Symphony & Orchestral',
        'thumbnail_url': 'https://i.ytimg.com/vi/i2b568P3Yqg/hqdefault.jpg',
        'lyrics_snippet': 'Sultani aabhyalat tula paahun sajani... Masterpiece orchestral romantic ballad recorded live with the Hollywood Symphony Orchestra.'
    },
    {
        'title': 'Apsara Aali',
        'movie_title': 'Natarang',
        'release_year': 2010,
        'era': '2010s Blockbusters',
        'singer': 'Bela Shende, Ajay-Atul',
        'composer': 'Ajay-Atul',
        'lyricist': 'Guru Thakur',
        'mood': 'Classical & Folk, High Energy & Dance',
        'thumbnail_url': 'https://i.ytimg.com/vi/y3V5fJ55g3s/hqdefault.jpg',
        'lyrics_snippet': 'Apsara aali indrapuritun aali... Legendary Lavani dance celebration featuring Sonalee Kulkarni with traditional Dholak beats.'
    },
    {
        'title': 'Ved Lavlay',
        'movie_title': 'Ved',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'singer': 'Ajay Gogavale, Vishal Dadlani',
        'composer': 'Ajay-Atul',
        'lyricist': 'Ajay-Atul',
        'mood': 'High Energy & Dance, Upbeat',
        'thumbnail_url': 'https://i.ytimg.com/vi/xL9j0K6K7L4/hqdefault.jpg',
        'lyrics_snippet': 'Aamhi premacha nachanar naara lavla... Energetic party anthem featuring Salman Khan and Riteish Deshmukh.'
    },
    {
        'title': 'Sur Niragas Ho',
        'movie_title': 'Katyar Kaljat Ghusali',
        'release_year': 2015,
        'era': '2010s Blockbusters',
        'singer': 'Shankar Mahadevan, Anandi Joshi',
        'composer': 'Shankar-Ehsaan-Loy, Pt. Jitendra Abhisheki',
        'lyricist': 'Mangesh Kangane',
        'mood': 'Classical & Semi-Classical, Devotional & Divine',
        'thumbnail_url': 'https://i.ytimg.com/vi/cQJ-k944Kqg/hqdefault.jpg',
        'lyrics_snippet': 'Sur niragas ho ganesharaya... Pure divine classical prayer to Lord Ganesha celebrating the sanctity of music.'
    },

    # --- South Indian Pan-Indian Hits ---
    {
        'title': 'Arabic Kuthu (Halamithi Habibo)',
        'movie_title': 'Beast',
        'release_year': 2022,
        'era': 'Latest (2020-2026)',
        'singer': 'Anirudh Ravichander, Jonita Gandhi',
        'composer': 'Anirudh Ravichander',
        'lyricist': 'Sivakarthikeyan',
        'mood': 'High Energy & Dance, Pop, Viral Dance',
        'thumbnail_url': 'https://i.ytimg.com/vi/KUN5Uf9mObQ/hqdefault.jpg',
        'lyrics_snippet': 'Halamithi habibo... Record-breaking global dance smash hit with Thalapathy Vijay and Pooja Hegde.'
    },
    {
        'title': 'Hukum (Thalaivar Alappara)',
        'movie_title': 'Jailer',
        'release_year': 2023,
        'era': 'Latest (2020-2026)',
        'singer': 'Anirudh Ravichander',
        'composer': 'Anirudh Ravichander',
        'lyricist': 'Super Subu',
        'mood': 'High Energy, Mass & Swag, Rock & Acoustic',
        'thumbnail_url': 'https://i.ytimg.com/vi/1F3hm6MfR1k/hqdefault.jpg',
        'lyrics_snippet': 'Hukum tiger ka hukum... Electrifying swagger anthem celebrating Superstar Rajinikanth with thumping rock beats.'
    },
    {
        'title': 'Enjoy Enjaami',
        'movie_title': 'Maajja Indie Singles',
        'release_year': 2021,
        'era': 'Latest (2020-2026)',
        'singer': 'Dhee, Arivu, Santhosh Narayanan',
        'composer': 'Santhosh Narayanan',
        'lyricist': 'Arivu',
        'mood': 'Folk Fusion, Hip Hop & Rap, Indian Indie',
        'thumbnail_url': 'https://i.ytimg.com/vi/eYq7WapuDLU/hqdefault.jpg',
        'lyrics_snippet': 'Enjoy enjaami vaango vaango onnagi... Celebrated Tamil folk-hip-hop crossover honoring ancestral earth and unity.'
    },
    {
        'title': 'Samajavaragamana',
        'movie_title': 'Ala Vaikunthapurramuloo',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'Sid Sriram',
        'composer': 'Thaman S',
        'lyricist': 'Sirivennela Seetharama Sastry',
        'mood': 'Romantic & Soulful, Carnatic Fusion',
        'thumbnail_url': 'https://i.ytimg.com/vi/ocpeD_U_rEE/hqdefault.jpg',
        'lyrics_snippet': 'Samajavaragamana ninu choosi aaga galana... Mesmerizing classical-contemporary romantic marvel sung by Sid Sriram.'
    },
    {
        'title': 'Butta Bomma',
        'movie_title': 'Ala Vaikunthapurramuloo',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'Armaan Malik',
        'composer': 'Thaman S',
        'lyricist': 'Ramajogayya Sastry',
        'mood': 'Upbeat & Cheerful, Pop, Dance',
        'thumbnail_url': 'https://i.ytimg.com/vi/2mDCVzruYzQ/hqdefault.jpg',
        'lyrics_snippet': 'Inchu minchu aravai kilo... Global viral phenomenon with effortless hook steps featuring Allu Arjun and Pooja Hegde.'
    },

    # --- English & Global Pop Hits ---
    {
        'title': 'Cruel Summer',
        'movie_title': 'Lover',
        'release_year': 2019,
        'era': '2010s Blockbusters',
        'singer': 'Taylor Swift',
        'composer': 'Taylor Swift, Jack Antonoff, St. Vincent',
        'lyricist': 'Taylor Swift',
        'mood': 'Global Pop, Synthpop, High Energy',
        'thumbnail_url': 'https://i.ytimg.com/vi/ic8j13U_FS8/hqdefault.jpg',
        'lyrics_snippet': 'Devils roll the dice angels roll their eyes... Global number-one synth-pop masterwork by Taylor Swift.'
    },
    {
        'title': 'Blinding Lights',
        'movie_title': 'After Hours',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'The Weeknd',
        'composer': 'The Weeknd, Max Martin, Oscar Holter',
        'lyricist': 'The Weeknd',
        'mood': 'Global Pop, Synthwave, Dance & Electronic',
        'thumbnail_url': 'https://i.ytimg.com/vi/4NRXx6U8ABQ/hqdefault.jpg',
        'lyrics_snippet': 'I said ooh I am blinded by the lights... The Billboard all-time number one synthwave hit celebrated worldwide.'
    },
    {
        'title': 'Levitating',
        'movie_title': 'Future Nostalgia',
        'release_year': 2020,
        'era': 'Latest (2020-2026)',
        'singer': 'Dua Lipa',
        'composer': 'Dua Lipa, Clarence Coffee Jr., Sarah Hudson',
        'lyricist': 'Dua Lipa',
        'mood': 'Global Pop, Disco, Dance & Electronic',
        'thumbnail_url': 'https://i.ytimg.com/vi/TUVcZfQe-Kw/hqdefault.jpg',
        'lyrics_snippet': 'If you wanna run away with me I know a galaxy... Infectious future-disco anthem with groove-heavy basslines.'
    },
    {
        'title': 'Shape of You',
        'movie_title': 'Divide',
        'release_year': 2017,
        'era': '2010s Blockbusters',
        'singer': 'Ed Sheeran',
        'composer': 'Ed Sheeran, Steve Mac, Johnny McDaid',
        'lyricist': 'Ed Sheeran',
        'mood': 'Global Pop, Tropical Pop, Upbeat',
        'thumbnail_url': 'https://i.ytimg.com/vi/JGwWNGJdvx8/hqdefault.jpg',
        'lyrics_snippet': 'The club isn\'t the best place to find a lover so the bar is where I go... Unprecedented global chart-topper by Ed Sheeran.'
    },
    {
        'title': 'Yellow',
        'movie_title': 'Parachutes',
        'release_year': 2000,
        'era': '2000s New Wave',
        'singer': 'Coldplay, Chris Martin',
        'composer': 'Coldplay',
        'lyricist': 'Chris Martin',
        'mood': 'Rock & Acoustic, Global Pop, Soulful & Melancholy',
        'thumbnail_url': 'https://i.ytimg.com/vi/yKNxeF4KMsY/hqdefault.jpg',
        'lyrics_snippet': 'Look at the stars look how they shine for you... Legendary alternative rock love declaration by Coldplay.'
    },
    {
        'title': 'Birds of a Feather',
        'movie_title': 'Hit Me Hard and Soft',
        'release_year': 2024,
        'era': 'Latest (2020-2026)',
        'singer': 'Billie Eilish',
        'composer': 'Billie Eilish, Finneas O\'Connell',
        'lyricist': 'Billie Eilish',
        'mood': 'Global Pop, Indie Pop, Romantic & Soulful',
        'thumbnail_url': 'https://i.ytimg.com/vi/d5gf9dXbPi0/hqdefault.jpg',
        'lyrics_snippet': 'I want you to stay till I\'m in the grave... Global viral pop sensation exploring boundless devotion and eternal love.'
    },
    {
        'title': 'Sunflower',
        'movie_title': 'Spider-Man: Into the Spider-Verse',
        'release_year': 2018,
        'era': '2010s Blockbusters',
        'singer': 'Post Malone, Swae Lee',
        'composer': 'Post Malone, Swae Lee, Louis Bell, Carter Lang',
        'lyricist': 'Post Malone, Swae Lee',
        'mood': 'Hip Hop & Rap, Global Pop, Chill Vibes',
        'thumbnail_url': 'https://i.ytimg.com/vi/ApXoWvfEYVU/hqdefault.jpg',
        'lyrics_snippet': 'Needless to say I keep her in check she was all bad-bad nevertheless... Diamond-certified breezy hip-hop pop phenomenon.'
    }
]

def main():
    print("=" * 60)
    print("Starting Comprehensive Dataset Enrichment...")
    print("=" * 60)

    # 1. Enrich Movies
    df_movies = pd.read_csv(MOVIES_CSV)
    print(f"Existing movies count: {len(df_movies)}")

    existing_movie_titles = set(df_movies['title'].astype(str).str.strip().str.lower())
    new_movies = []
    
    for m in EXPANDED_MOVIES:
        if m['title'].strip().lower() not in existing_movie_titles:
            m['movie_id'] = f"mov_{len(df_movies) + len(new_movies) + 1:05d}"
            new_movies.append(m)
            existing_movie_titles.add(m['title'].strip().lower())

    if new_movies:
        df_new_movies = pd.DataFrame(new_movies)
        df_movies = pd.concat([df_movies, df_new_movies], ignore_index=True)
        df_movies.to_csv(MOVIES_CSV, index=False)
        print(f"-> Added {len(new_movies)} diverse new movies. Total now: {len(df_movies)}")
    else:
        print("-> All movies already present in dataset.")

    # 2. Enrich Songs
    df_songs = pd.read_csv(SONGS_CSV)
    print(f"Existing songs count: {len(df_songs)}")

    existing_song_titles = set(df_songs['title'].astype(str).str.strip().str.lower())
    new_songs = []

    for s in EXPANDED_SONGS:
        if s['title'].strip().lower() not in existing_song_titles:
            s['song_id'] = f"sng_{len(df_songs) + len(new_songs) + 1:05d}"
            new_songs.append(s)
            existing_song_titles.add(s['title'].strip().lower())

    if new_songs:
        df_new_songs = pd.DataFrame(new_songs)
        df_songs = pd.concat([df_songs, df_new_songs], ignore_index=True)
        df_songs.to_csv(SONGS_CSV, index=False)
        print(f"-> Added {len(new_songs)} diverse new songs (Indie, Marathi, South, English). Total now: {len(df_songs)}")
    else:
        print("-> All songs already present in dataset.")

    print("\n=" * 60)
    print("Dataset Enrichment Successfully Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
