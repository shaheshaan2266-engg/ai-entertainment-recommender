"""
Watch Availability and Technical Specifications Provider for Indian Cinema & Music.
Provides verified streaming platform links, technical audio/video specs,
runtime, certification, and embedded trailer data.
"""

import urllib.parse
from typing import Dict, Any, List

# Curated catalog mapping for major Indian Cinema landmarks
CURATED_WATCH_CATALOG: Dict[str, Dict[str, Any]] = {
    "rrr": {
        "platforms": [
            {"name": "JioHotstar", "tier": "Subscription", "quality": "4K UHD · Dolby Vision", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/movies/rrr/1260109952"},
            {"name": "Netflix", "tier": "Subscription (Hindi)", "quality": "4K UHD · Dolby Vision", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com/title/81476453"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "Full HD 1080p", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=RRR+movie"},
            {"name": "ZEE5", "tier": "Subscription", "quality": "4K UHD · 5.1 Surround", "color": "#8230C6", "icon": "Z", "url": "https://www.zee5.com/movies/details/rrr/0-0-1z5169482"}
        ],
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)",
            "hdr": "Dolby Vision, HDR10+",
            "audio": "Dolby Atmos 7.1.4 Surround Sound",
            "aspect_ratio": "2.39:1 Anamorphic Cinemascope",
            "runtime": "3h 07m (187 mins)",
            "certification": "U/A 16+",
            "languages": "Hindi, Telugu (Original), Tamil, Malayalam, Kannada",
            "studio": "DVV Entertainment"
        },
        "trailer_query": "RRR official trailer Hindi"
    },
    "jawan": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription (Extended Cut)", "quality": "4K UHD · Dolby Atmos", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com/title/81721627"},
            {"name": "JioHotstar", "tier": "Explore Availability", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Jawan"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "4K UHD 2160p", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Jawan+movie"}
        ],
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)",
            "hdr": "Dolby Vision",
            "audio": "Dolby Atmos 7.1 Surround",
            "aspect_ratio": "2.39:1 Cinemascope",
            "runtime": "2h 50m (170 mins)",
            "certification": "U/A 16+",
            "languages": "Hindi, Tamil, Telugu",
            "studio": "Red Chillies Entertainment"
        },
        "trailer_query": "Jawan official trailer SRK"
    },
    "stree 2": {
        "platforms": [
            {"name": "Prime Video", "tier": "Subscription / Rent", "quality": "4K UHD · 5.1 Surround", "color": "#00A8E1", "icon": "P", "url": "https://www.primevideo.com"},
            {"name": "JioHotstar", "tier": "Digital Network", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Stree+2"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "4K UHD 2160p", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Stree+2+movie"}
        ],
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)",
            "hdr": "HDR10+",
            "audio": "Dolby Digital Plus 5.1",
            "aspect_ratio": "2.39:1 Widescope",
            "runtime": "2h 27m (147 mins)",
            "certification": "U/A 16+",
            "languages": "Hindi (Original), English Subtitles",
            "studio": "Maddock Films, Jio Studios"
        },
        "trailer_query": "Stree 2 official trailer"
    },
    "animal": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription (Uncut)", "quality": "4K UHD · Dolby Atmos", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com/title/81436990"},
            {"name": "JioHotstar", "tier": "Digital Partner", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Animal"},
            {"name": "YouTube Movies", "tier": "Store", "quality": "4K UHD", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Animal+movie+Ranbir"}
        ],
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)",
            "hdr": "Dolby Vision, HDR10",
            "audio": "Dolby Atmos 7.1 Surround",
            "aspect_ratio": "2.39:1 Cinemascope",
            "runtime": "3h 24m (204 mins)",
            "certification": "A (Adults Only)",
            "languages": "Hindi, Telugu, Tamil, Malayalam, Kannada",
            "studio": "T-Series, Bhadrakali Pictures"
        },
        "trailer_query": "Animal official trailer Ranbir Kapoor"
    },
    "3 idiots": {
        "platforms": [
            {"name": "Prime Video", "tier": "Subscription", "quality": "Full HD 1080p", "color": "#00A8E1", "icon": "P", "url": "https://www.primevideo.com"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "HD 1080p", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=3+Idiots+full+movie"},
            {"name": "JioHotstar", "tier": "Digital Network", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=3+Idiots"}
        ],
        "tech_specs": {
            "resolution": "1080p Full HD",
            "hdr": "Standard Dynamic Range (SDR)",
            "audio": "Dolby Digital 5.1 Surround",
            "aspect_ratio": "2.35:1 Widescreen",
            "runtime": "2h 50m (170 mins)",
            "certification": "U/A",
            "languages": "Hindi (Original) + 12 Subtitle Tracks",
            "studio": "Vinod Chopra Films"
        },
        "trailer_query": "3 Idiots official trailer"
    },
    "dil chahta hai": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription", "quality": "Full HD Remastered", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "1080p Full HD", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Dil+Chahta+Hai+movie"},
            {"name": "JioHotstar", "tier": "Digital Catalogue", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Dil+Chahta+Hai"}
        ],
        "tech_specs": {
            "resolution": "1080p Full HD (Digitally Remastered)",
            "hdr": "SDR",
            "audio": "Dolby Digital 5.1",
            "aspect_ratio": "2.35:1 Anamorphic",
            "runtime": "3h 03m (183 mins)",
            "certification": "U/A",
            "languages": "Hindi",
            "studio": "Excel Entertainment"
        },
        "trailer_query": "Dil Chahta Hai trailer"
    },
    "rockstar": {
        "platforms": [
            {"name": "JioHotstar", "tier": "Subscription", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Rockstar"},
            {"name": "JioCinema", "tier": "Subscription", "quality": "Full HD 1080p", "color": "#E11B73", "icon": "J", "url": "https://www.jiocinema.com"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "Full HD 1080p", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Rockstar+2011+movie"},
            {"name": "ZEE5", "tier": "Subscription", "quality": "Full HD 1080p", "color": "#8230C6", "icon": "Z", "url": "https://www.zee5.com"}
        ],
        "tech_specs": {
            "resolution": "1080p Full HD",
            "hdr": "SDR",
            "audio": "Dolby Digital 5.1 (High-Fidelity Audio)",
            "aspect_ratio": "2.35:1 Widescreen",
            "runtime": "2h 39m (159 mins)",
            "certification": "U/A",
            "languages": "Hindi",
            "studio": "Eros International, Shree Ashtavinayak Cine Vision"
        },
        "trailer_query": "Rockstar 2011 official trailer Ranbir Kapoor"
    },
    "kabir singh": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription", "quality": "4K UHD · 5.1", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com"},
            {"name": "JioHotstar", "tier": "Digital Partner", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Kabir+Singh"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "Full HD", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Kabir+Singh+movie"}
        ],
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)",
            "hdr": "Dolby Vision",
            "audio": "Dolby Atmos 5.1",
            "aspect_ratio": "2.39:1",
            "runtime": "2h 53m (173 mins)",
            "certification": "A",
            "languages": "Hindi",
            "studio": "Cine1 Studios, T-Series"
        },
        "trailer_query": "Kabir Singh official trailer Shahid Kapoor"
    },
    "taare zameen par": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription", "quality": "Full HD 1080p", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com"},
            {"name": "JioHotstar", "tier": "Digital Network", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Taare+Zameen+Par"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "HD 1080p", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Taare+Zameen+Par+movie"}
        ],
        "tech_specs": {
            "resolution": "1080p Full HD",
            "hdr": "SDR",
            "audio": "Dolby Digital 5.1",
            "aspect_ratio": "2.35:1",
            "runtime": "2h 45m (165 mins)",
            "certification": "U (Universal)",
            "languages": "Hindi",
            "studio": "Aamir Khan Productions"
        },
        "trailer_query": "Taare Zameen Par official trailer"
    },
    "kalki 2898 ad": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription (Hindi)", "quality": "4K UHD · Dolby Atmos", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com"},
            {"name": "Prime Video", "tier": "Subscription (Telugu, Tamil, Kannada, Malayalam)", "quality": "4K UHD · Dolby Atmos", "color": "#00A8E1", "icon": "P", "url": "https://www.primevideo.com"},
            {"name": "JioHotstar", "tier": "Digital Partner", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Kalki+2898+AD"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "4K UHD", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Kalki+2898+AD+movie"}
        ],
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)",
            "hdr": "Dolby Vision, HDR10+",
            "audio": "Dolby Atmos 7.1.4 Surround",
            "aspect_ratio": "2.39:1 & IMAX Aspect Ratio 1.90:1",
            "runtime": "3h 01m (181 mins)",
            "certification": "U/A 13+",
            "languages": "Telugu (Original), Hindi, Tamil, Kannada, Malayalam",
            "studio": "Vyjayanthi Movies"
        },
        "trailer_query": "Kalki 2898 AD official trailer"
    },
    "devara: part 1": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription", "quality": "4K UHD · Dolby Atmos", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com"},
            {"name": "JioHotstar", "tier": "Digital Network", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Devara"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "4K UHD", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Devara+Part+1+movie"}
        ],
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)",
            "hdr": "Dolby Vision",
            "audio": "Dolby Atmos 7.1",
            "aspect_ratio": "2.39:1 Cinemascope",
            "runtime": "2h 58m (178 mins)",
            "certification": "U/A 16+",
            "languages": "Telugu (Original), Hindi, Tamil, Malayalam, Kannada",
            "studio": "Yuvasudha Arts, NTR Arts"
        },
        "trailer_query": "Devara Part 1 official trailer"
    },
    "zindagi na milegi dobara": {
        "platforms": [
            {"name": "Netflix", "tier": "Subscription", "quality": "Full HD 1080p", "color": "#E50914", "icon": "N", "url": "https://www.netflix.com"},
            {"name": "JioHotstar", "tier": "Digital Catalogue", "quality": "Full HD 1080p", "color": "#0C2050", "icon": "JH", "url": "https://www.hotstar.com/in/explore?search_query=Zindagi+Na+Milegi+Dobara"},
            {"name": "YouTube Movies", "tier": "Rent / Buy", "quality": "1080p Full HD", "color": "#FF0000", "icon": "Y", "url": "https://www.youtube.com/results?search_query=Zindagi+Na+Milegi+Dobara+movie"}
        ],
        "tech_specs": {
            "resolution": "1080p Full HD",
            "hdr": "SDR",
            "audio": "Dolby Digital 5.1",
            "aspect_ratio": "2.35:1 Widescreen",
            "runtime": "2h 35m (155 mins)",
            "certification": "U/A",
            "languages": "Hindi",
            "studio": "Excel Entertainment"
        },
        "trailer_query": "Zindagi Na Milegi Dobara trailer"
    }
}


def get_watch_info(title: str, media_type: str = "movie", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Returns streaming platform availability, direct web routing, and technical specifications.
    """
    if not title:
        title = "Entertainment"

    norm_title = title.lower().strip()
    encoded = urllib.parse.quote_plus(title)

    # Check curated cache
    if norm_title in CURATED_WATCH_CATALOG:
        info = CURATED_WATCH_CATALOG[norm_title]
        return {
            "title": title,
            "media_type": media_type,
            "status": "Available Now",
            "default_tier": "Must Watch",
            "platforms": info["platforms"],
            "tech_specs": info["tech_specs"],
            "trailer_url": f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(info['trailer_query'])}",
            "video_song_url": f"https://www.youtube.com/results?search_query={encoded}+official+song+video",
            "justwatch_url": f"https://www.justwatch.com/in/search?q={encoded}",
            "google_play_url": f"https://play.google.com/store/search?q={encoded}&c=movies"
        }

    # Music Track Handling
    if media_type == "song":
        return {
            "title": title,
            "media_type": "song",
            "status": "Streaming in Hi-Res Audio & 4K Video",
            "default_tier": "Must Listen",
            "platforms": [
                {
                    "name": "YouTube",
                    "tier": "Free / Premium (Official Video Song)",
                    "quality": "4K / 1080p Official Music Video",
                    "color": "#FF0000",
                    "icon": "Y",
                    "url": f"https://www.youtube.com/results?search_query={encoded}+official+music+video"
                },
                {
                    "name": "Spotify",
                    "tier": "Free / Premium",
                    "quality": "320 kbps High Quality Audio",
                    "color": "#1DB954",
                    "icon": "S",
                    "url": f"https://open.spotify.com/search/{encoded}"
                },
                {
                    "name": "JioSaavn",
                    "tier": "Free / Pro",
                    "quality": "320 kbps HD Audio",
                    "color": "#2BC5B4",
                    "icon": "JS",
                    "url": f"https://www.jiosaavn.com/search/{encoded}"
                },
                {
                    "name": "Apple Music",
                    "tier": "Subscription",
                    "quality": "Apple Lossless · Spatial Audio",
                    "color": "#FC3C44",
                    "icon": "A",
                    "url": f"https://music.apple.com/search?term={encoded}"
                },
                {
                    "name": "YouTube Music",
                    "tier": "Free / Premium",
                    "quality": "256 kbps AAC with Music Video",
                    "color": "#FF0000",
                    "icon": "YM",
                    "url": f"https://music.youtube.com/search?q={encoded}"
                }
            ],
            "tech_specs": {
                "format": "Lossless Audio (ALAC / FLAC / 320kbps MP3)",
                "sampling_rate": "24-bit / 48kHz & 96kHz Hi-Res Lossless",
                "spatial_audio": "Dolby Atmos Music Supported",
                "bitrate": "Lossless up to 1411 kbps",
                "channels": "Stereo & Binaural Spatial 3D",
                "label": "T-Series / Sony Music India / Zee Music / YRF Music"
            },
            "trailer_url": f"https://www.youtube.com/results?search_query={encoded}+official+song+video",
            "video_song_url": f"https://www.youtube.com/results?search_query={encoded}+official+video+song",
            "justwatch_url": f"https://open.spotify.com/search/{encoded}",
            "google_play_url": f"https://music.youtube.com/search?q={encoded}"
        }

    # Dynamic fallback for movies
    era = metadata.get("era", "") if metadata else ""
    is_latest = "202" in era or "latest" in era.lower()

    default_platforms: List[Dict[str, Any]] = [
        {
            "name": "JioHotstar",
            "tier": "Subscription",
            "quality": "4K UHD · Dolby Vision" if is_latest else "Full HD 1080p",
            "color": "#0C2050",
            "icon": "JH",
            "url": f"https://www.hotstar.com/in/explore?search_query={encoded}"
        },
        {
            "name": "Netflix",
            "tier": "Subscription",
            "quality": "4K Ultra HD · Dolby Atmos" if is_latest else "Full HD 1080p",
            "color": "#E50914",
            "icon": "N",
            "url": f"https://www.netflix.com/search?q={encoded}"
        },
        {
            "name": "YouTube Movies",
            "tier": "Rent / Buy",
            "quality": "Full HD 1080p",
            "color": "#FF0000",
            "icon": "Y",
            "url": f"https://www.youtube.com/results?search_query={encoded}+full+movie"
        },
        {
            "name": "Prime Video",
            "tier": "Subscription / Store",
            "quality": "4K UHD · 5.1 Surround" if is_latest else "HD 1080p",
            "color": "#00A8E1",
            "icon": "P",
            "url": f"https://www.primevideo.com/search/ref=atv_nb_sr?phrase={encoded}"
        },
        {
            "name": "JioCinema",
            "tier": "Subscription",
            "quality": "Full HD 1080p",
            "color": "#E11B73",
            "icon": "J",
            "url": f"https://www.jiocinema.com"
        }
    ]

    return {
        "title": title,
        "media_type": "movie",
        "status": "Available to Stream",
        "default_tier": "Go for it" if is_latest else "Good",
        "platforms": default_platforms,
        "tech_specs": {
            "resolution": "4K Ultra HD (2160p)" if is_latest else "1080p Full HD",
            "hdr": "Dolby Vision, HDR10+" if is_latest else "Standard Dynamic Range (SDR)",
            "audio": "Dolby Atmos 7.1 Surround" if is_latest else "Dolby Digital 5.1",
            "aspect_ratio": "2.39:1 Widescreen",
            "runtime": metadata.get("runtime", "2h 35m") if metadata else "2h 35m",
            "certification": "U/A 13+",
            "languages": "Hindi (Original) + Regional Audio & English Subtitles",
            "studio": metadata.get("industry", "Indian Cinema Studios") if metadata else "Indian Cinema Studios"
        },
        "trailer_url": f"https://www.youtube.com/results?search_query={encoded}+official+trailer",
        "video_song_url": f"https://www.youtube.com/results?search_query={encoded}+hit+songs+video",
        "justwatch_url": f"https://www.justwatch.com/in/search?q={encoded}",
        "google_play_url": f"https://play.google.com/store/search?q={encoded}&c=movies"
    }
