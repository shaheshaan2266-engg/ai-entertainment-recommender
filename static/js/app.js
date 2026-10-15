/**
 * Cinevibe AI — Cyber-Cinema Experience Application Logic
 * Implements sidebar tab switching, light highlight pop-up animations,
 * real-time mode filtering, Discover Hub, My List / Favorites storage,
 * and Knowledge Graph relational chaining.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Navigation & Tabs
  const navItems = document.querySelectorAll(".nav-item");
  const navHome = document.getElementById("nav-home");
  const navDiscover = document.getElementById("nav-discover");
  const navGraph = document.getElementById("nav-graph");
  const navMyList = document.getElementById("nav-mylist");
  const navFavorites = document.getElementById("nav-favorites");

  const badgeMyList = document.getElementById("badge-my-list");
  const badgeFavorites = document.getElementById("badge-favorites");

  // Filter Pills
  const filterPills = document.querySelectorAll(".filter-pill");

  // Main Sections
  const mainContent = document.getElementById("main-content");
  const heroBanner = document.getElementById("hero-banner");
  const searchSection = document.querySelector(".search-section");
  const searchResultsSection = document.getElementById("search-results-section");
  const searchResultsGrid = document.getElementById("search-results-grid");
  const sectionTrending = document.getElementById("section-trending");
  const sectionEntity = document.getElementById("section-entity");
  const sectionDual = document.getElementById("section-dual");
  const sectionDiscover = document.getElementById("section-discover");

  // Search Elements
  const searchInput = document.getElementById("search-input");
  const searchClearBtn = document.getElementById("search-clear-btn");
  const searchMicBtn = document.getElementById("search-mic-btn");
  const autocompletePanel = document.getElementById("autocomplete-panel");
  const autocompleteItems = document.getElementById("autocomplete-items");

  const resultsBadge = document.getElementById("results-badge");
  const queryHighlight = document.getElementById("query-highlight");
  const resultsDetail = document.getElementById("results-detail");
  const exploreChainTopBtn = document.getElementById("explore-chain-top-btn");
  const closeResultsBtn = document.getElementById("close-results-btn");

  // Multi-Genre & Year Filter Elements
  const searchFiltersToggleBtn = document.getElementById("search-filters-toggle-btn");
  const activeFiltersCount = document.getElementById("active-filters-count");
  const activeFiltersBar = document.getElementById("active-filters-bar");
  const activeFilterChipsList = document.getElementById("active-filter-chips-list");
  const clearAllFiltersBtn = document.getElementById("clear-all-filters-btn");
  const searchFilterPanel = document.getElementById("search-filter-panel");
  const filterModeIndicator = document.getElementById("filter-mode-indicator");
  const genreHintText = document.getElementById("genre-hint-text");
  const filterResetBtn = document.getElementById("filter-reset-btn");
  const filterCloseBtn = document.getElementById("filter-close-btn");
  const genreChipsContainer = document.getElementById("genre-chips-container");
  const yearPresetsChips = document.getElementById("year-presets-chips");
  const exactYearSelect = document.getElementById("exact-year-select");
  const activeTagsSummary = document.getElementById("active-tags-summary");
  const applyFiltersBtn = document.getElementById("apply-filters-btn");

  // Carousels
  const trendingTrack = document.getElementById("trending-carousel");
  const btnTrendingNext = document.getElementById("btn-trending-next");
  const entityTrack = document.getElementById("entity-carousel");
  const btnEntityNext = document.getElementById("btn-entity-next");

  // Graph Modal
  const chainModalOverlay = document.getElementById("chain-modal-overlay");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalBadge = document.getElementById("modal-badge");
  const modalName = document.getElementById("modal-name");
  const modalStats = document.getElementById("modal-stats");
  const modalCollabsList = document.getElementById("modal-collabs-list");
  const modalMoviesList = document.getElementById("modal-movies-list");
  const modalSongsList = document.getElementById("modal-songs-list");

  // List / Favorites Drawer
  const listDrawerOverlay = document.getElementById("list-drawer-overlay");
  const drawerCloseBtn = document.getElementById("drawer-close-btn");
  const drawerClearBtn = document.getElementById("drawer-clear-btn");
  const drawerBadge = document.getElementById("drawer-badge");
  const drawerTitle = document.getElementById("drawer-title");
  const drawerItemsList = document.getElementById("drawer-items-list");

  // Discover Controls
  const eraChips = document.querySelectorAll(".era-chip");
  const genreChips = document.querySelectorAll(".genre-chip");
  const discoverResultsGrid = document.getElementById("discover-results-grid");

  // Audio Toast & Back to Top
  const audioToast = document.getElementById("audio-toast");
  const toastTitle = document.getElementById("toast-title");
  const toastArtist = document.getElementById("toast-artist");
  const toastPlayLink = document.getElementById("toast-play-link");
  const toastClose = document.getElementById("toast-close");
  const floatingTopBtn = document.getElementById("floating-top-btn");

  // Watch Availability & Technical Specs Modal Elements
  const watchModalOverlay = document.getElementById("watch-modal-overlay");
  const watchModalDialog = document.getElementById("watch-modal-dialog");
  const watchCloseBtn = document.getElementById("watch-close-btn");
  const watchModalTitle = document.getElementById("watch-modal-title");
  const watchModalSubtitle = document.getElementById("watch-modal-subtitle");
  const watchCertBadge = document.getElementById("watch-cert-badge");
  const watchResBadge = document.getElementById("watch-res-badge");
  const watchAudioBadge = document.getElementById("watch-audio-badge");
  const watchRatingBadge = document.getElementById("watch-rating-badge");
  const streamingAvailabilityStatus = document.getElementById("streaming-availability-status");
  const platformsGrid = document.getElementById("platforms-grid");
  const linkJustwatch = document.getElementById("link-justwatch");
  const linkGoogleplay = document.getElementById("link-googleplay");
  const techSpecsGrid = document.getElementById("tech-specs-grid");
  const launchTrailerBtn = document.getElementById("launch-trailer-btn");
  const soundtrackList = document.getElementById("soundtrack-list");
  const watchTabBtns = document.querySelectorAll(".watch-tab-btn");
  const watchActRecommend = document.getElementById("watch-act-recommend");
  const watchActGraph = document.getElementById("watch-act-graph");
  const watchActSave = document.getElementById("watch-act-save");
  const watchSaveText = document.getElementById("watch-save-text");
  const watchActShare = document.getElementById("watch-act-share");

  // Floating Rating Popover Elements
  const cardRatingPopover = document.getElementById("card-rating-popover");
  const popoverTargetTitle = document.getElementById("popover-target-title");
  const popoverCloseBtn = document.getElementById("popover-close-btn");
  const popTierMustText = document.getElementById("pop-tier-must-text");

  let activePopoverTitle = "";
  let activePopoverType = "movie";

  let currentWatchTitle = "";
  let currentWatchType = "movie";
  let currentWatchData = null;

  // HTML escaping utility
  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Audience Vibe Rating System (Tiers: Bad, OK-ish, Good, Go for it, Must Watch / Listen)
  const RATING_TIERS = {
    bad: {
      key: "bad",
      movieLabel: "Bad",
      songLabel: "Bad",
      level: 1,
      color: "#ef4444",
      desc: "Skip it"
    },
    "ok-ish": {
      key: "ok-ish",
      movieLabel: "OK-ish",
      songLabel: "OK-ish",
      level: 2,
      color: "#f59e0b",
      desc: "Timepass"
    },
    good: {
      key: "good",
      movieLabel: "Good",
      songLabel: "Good",
      level: 3,
      color: "#10b981",
      desc: "Enjoyable"
    },
    "go-for-it": {
      key: "go-for-it",
      movieLabel: "Go for it",
      songLabel: "Go for it",
      level: 4,
      color: "#06b6d4",
      desc: "Recommended"
    },
    must: {
      key: "must",
      movieLabel: "Must Watch",
      songLabel: "Must Listen",
      level: 5,
      color: "#a855f7",
      desc: "Masterpiece"
    }
  };

  const DEFAULT_RATINGS = {
    "rrr": "must",
    "jawan": "go-for-it",
    "stree 2": "go-for-it",
    "animal": "go-for-it",
    "3 idiots": "must",
    "dil chahta hai": "must",
    "rockstar": "must",
    "kabir singh": "good",
    "taare zameen par": "must",
    "kalki 2898 ad": "go-for-it",
    "devara: part 1": "good",
    "zindagi na milegi dobara": "must",
    "naatu naatu": "must",
    "tauba tauba": "go-for-it",
    "aaj ki raat": "go-for-it",
    "kesariya": "must",
    "oo antava mava oo oo antava": "go-for-it",
    "apna bana le": "must",
    "kaavaalaa": "go-for-it",
    "pehle bhi main": "must",
    "satranga": "must",
    "o maahi": "go-for-it",
    "kun faya kun": "must",
    "tum hi ho": "must",
    "agar tum saath ho": "must"
  };

  function getUserRatings() {
    try {
      return JSON.parse(localStorage.getItem("cinevibe_user_ratings") || localStorage.getItem("velora_user_ratings") || "{}");
    } catch (e) {
      return {};
    }
  }

  function getItemRating(title, mediaType = "movie") {
    const ratings = getUserRatings();
    const normTitle = (title || "").trim().toLowerCase();
    let tierKey = ratings[normTitle];
    if (!tierKey) {
      tierKey = DEFAULT_RATINGS[normTitle] || (mediaType === "song" ? "go-for-it" : "good");
    }
    const tier = RATING_TIERS[tierKey] || RATING_TIERS["good"];
    const isMovie = (mediaType || "movie").toLowerCase() === "movie";
    return {
      ...tier,
      label: isMovie ? tier.movieLabel : tier.songLabel,
      isUserRated: Boolean(ratings[normTitle])
    };
  }

  function saveItemRating(title, tierKey, mediaType = "movie") {
    if (!title) return;
    const ratings = getUserRatings();
    const normTitle = title.trim().toLowerCase();
    ratings[normTitle] = tierKey;
    localStorage.setItem("cinevibe_user_ratings", JSON.stringify(ratings));
    localStorage.setItem("velora_user_ratings", JSON.stringify(ratings));

    updateCardsRatingUI(title);
    const tier = RATING_TIERS[tierKey];
    const isMovie = (mediaType || "movie").toLowerCase() === "movie";
    const label = isMovie ? tier.movieLabel : tier.songLabel;
    showAudioToast(`Rated "${title}" as ${label}!`, "Rating Updated");
  }

  function updateCardsRatingUI(title) {
    if (!title) return;
    const normTitle = title.trim().toLowerCase();

    document.querySelectorAll(".poster-card").forEach(card => {
      if ((card.dataset.title || "").trim().toLowerCase() === normTitle) {
        const type = card.dataset.type || "movie";
        const rating = getItemRating(title, type);
        const ratingBtn = card.querySelector(".card-rating-meter");
        if (ratingBtn) {
          ratingBtn.title = `Audience Vibe Rating: ${rating.label} · Click to rate`;
          const dot = ratingBtn.querySelector(".rating-dot");
          if (dot) {
            dot.style.backgroundColor = rating.color;
            dot.style.boxShadow = `0 0 7px ${rating.color}`;
          }
          const lbl = ratingBtn.querySelector(".rating-label");
          if (lbl) lbl.textContent = rating.label;
          const bars = ratingBtn.querySelectorAll(".meter-bar-segment");
          bars.forEach((b, idx) => {
            b.classList.toggle("active", idx < rating.level);
          });
        }
      }
    });

    if (currentWatchTitle && currentWatchTitle.trim().toLowerCase() === normTitle) {
      renderWatchModalRating(currentWatchTitle, currentWatchType);
    }
  }

  function openRatingPopover(anchorEl, title, mediaType = "movie") {
    if (!cardRatingPopover) return;
    activePopoverTitle = title;
    activePopoverType = mediaType;

    if (popoverTargetTitle) {
      popoverTargetTitle.textContent = title;
    }
    if (popTierMustText) {
      popTierMustText.textContent = mediaType === "movie" ? "Must Watch" : "Must Listen";
    }

    const currentRating = getItemRating(title, mediaType);
    document.querySelectorAll(".popover-tier-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tier === currentRating.key);
    });

    const rect = anchorEl.getBoundingClientRect();
    const popoverWidth = 220;
    let left = rect.left + window.scrollX;
    let top = rect.bottom + window.scrollY + 6;

    if (left + popoverWidth > window.innerWidth - 10) {
      left = window.innerWidth - popoverWidth - 14;
    }
    if (left < 10) left = 10;

    cardRatingPopover.style.left = `${left}px`;
    cardRatingPopover.style.top = `${top}px`;
    cardRatingPopover.classList.remove("hidden");
  }

  function closeRatingPopover() {
    if (cardRatingPopover) {
      cardRatingPopover.classList.add("hidden");
    }
  }

  if (popoverCloseBtn) {
    popoverCloseBtn.addEventListener("click", closeRatingPopover);
  }

  document.querySelectorAll(".popover-tier-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tier = btn.dataset.tier;
      if (activePopoverTitle) {
        saveItemRating(activePopoverTitle, tier, activePopoverType);
      }
      closeRatingPopover();
    });
  });

  document.addEventListener("click", (e) => {
    if (cardRatingPopover && !cardRatingPopover.classList.contains("hidden")) {
      if (!e.target.closest("#card-rating-popover") && !e.target.closest(".card-rating-meter")) {
        closeRatingPopover();
      }
    }
  });

  // Modal Rating Strip Listeners
  document.querySelectorAll("#modal-tier-buttons .tier-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tier = btn.dataset.tier;
      if (currentWatchTitle) {
        saveItemRating(currentWatchTitle, tier, currentWatchType);
        renderWatchModalRating(currentWatchTitle, currentWatchType);
      }
    });
  });

  function renderWatchModalRating(title, type = "movie") {
    const ratingInfo = getItemRating(title, type);
    const isMovie = (type || "movie").toLowerCase() === "movie";

    const modalMeterBars = document.getElementById("modal-meter-bars");
    if (modalMeterBars) {
      modalMeterBars.innerHTML = "";
      for (let i = 1; i <= 5; i++) {
        const bar = document.createElement("span");
        bar.className = `m-bar ${i <= ratingInfo.level ? 'active' : ''}`;
        if (i <= ratingInfo.level) bar.style.backgroundColor = ratingInfo.color;
        modalMeterBars.appendChild(bar);
      }
    }

    const modalRatingDot = document.getElementById("modal-rating-dot");
    if (modalRatingDot) {
      modalRatingDot.style.backgroundColor = ratingInfo.color;
      modalRatingDot.style.boxShadow = `0 0 8px ${ratingInfo.color}`;
    }

    const modalRatingLabel = document.getElementById("modal-rating-label");
    if (modalRatingLabel) {
      modalRatingLabel.textContent = ratingInfo.label;
    }

    const modalMustText = document.getElementById("modal-must-text");
    if (modalMustText) {
      modalMustText.textContent = isMovie ? "Must Watch" : "Must Listen";
    }

    document.querySelectorAll("#modal-tier-buttons .tier-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tier === ratingInfo.key);
    });
  }

  // State Management
  let currentMode = "movies";
  let activeSeedTitle = "";
  let activeSeedType = "movie";
  let debounceTimer = null;
  let currentDrawerType = "list"; // 'list' or 'favorites'

  let selectedEra = "All";
  let selectedGenre = "All";

  // Local Storage Data
  let myList = JSON.parse(localStorage.getItem("cinevibe_my_list") || localStorage.getItem("velora_my_list") || "[]");
  let myFavorites = JSON.parse(localStorage.getItem("cinevibe_favorites") || localStorage.getItem("velora_favorites") || "[]");

  // Multi-Genre & Year Filter State
  const selectedMultiGenres = new Set();
  let selectedYearFilter = "All";
  let cachedFilterOptions = null;

  updateBadges();
  initFilterIntelligence();

  // -------------------------------------------------------------
  // 1. SIDEBAR NAVIGATION TABS FUNCTIONALITY (FULL IN-APP PAGES)
  // -------------------------------------------------------------
  function setActiveNavTab(activeEl) {
    navItems.forEach(item => item.classList.remove("active"));
    if (activeEl) {
      activeEl.classList.add("active");
    }
  }

  function navigateTo(route, updateHash = true) {
    const r = (route || "home").replace(/^#/, "").toLowerCase();

    // Reset all nav active states
    navItems.forEach(item => item.classList.remove("active"));

    if (r === "discover") {
      if (navDiscover) navDiscover.classList.add("active");
      if (mainContent) mainContent.setAttribute("data-page", "discover");
      chainModalOverlay.classList.add("hidden");
      listDrawerOverlay.classList.add("hidden");
      if (profileDrawerOverlay) profileDrawerOverlay.classList.add("hidden");
      sectionDiscover.classList.remove("hidden");
      searchResultsSection.classList.add("hidden");
      if (!discoverResultsGrid.children.length) {
        loadDiscoverContent();
      }
      if (updateHash && window.location.hash !== "#discover") {
        history.pushState(null, "", "#discover");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else if (r === "graph") {
      if (navGraph) navGraph.classList.add("active");
      if (mainContent) mainContent.setAttribute("data-page", "graph");
      listDrawerOverlay.classList.add("hidden");
      if (profileDrawerOverlay) profileDrawerOverlay.classList.add("hidden");
      sectionDiscover.classList.add("hidden");
      searchResultsSection.classList.add("hidden");
      openChainModal("actor", activeSeedTitle || "Shah Rukh Khan");
      if (updateHash && window.location.hash !== "#graph") {
        history.pushState(null, "", "#graph");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else if (r === "mylist" || r === "list") {
      if (navMyList) navMyList.classList.add("active");
      if (mainContent) mainContent.setAttribute("data-page", "list");
      chainModalOverlay.classList.add("hidden");
      if (profileDrawerOverlay) profileDrawerOverlay.classList.add("hidden");
      sectionDiscover.classList.add("hidden");
      searchResultsSection.classList.add("hidden");
      openDrawer("list");
      if (updateHash && window.location.hash !== "#mylist") {
        history.pushState(null, "", "#mylist");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else if (r === "favorites") {
      if (navFavorites) navFavorites.classList.add("active");
      if (mainContent) mainContent.setAttribute("data-page", "favorites");
      chainModalOverlay.classList.add("hidden");
      if (profileDrawerOverlay) profileDrawerOverlay.classList.add("hidden");
      sectionDiscover.classList.add("hidden");
      searchResultsSection.classList.add("hidden");
      openDrawer("favorites");
      if (updateHash && window.location.hash !== "#favorites") {
        history.pushState(null, "", "#favorites");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else if (r === "profile") {
      if (navProfile) navProfile.classList.add("active");
      if (mainContent) mainContent.setAttribute("data-page", "profile");
      chainModalOverlay.classList.add("hidden");
      listDrawerOverlay.classList.add("hidden");
      sectionDiscover.classList.add("hidden");
      searchResultsSection.classList.add("hidden");
      openProfileDrawer();
      if (updateHash && window.location.hash !== "#profile") {
        history.pushState(null, "", "#profile");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      // Home
      if (navHome) navHome.classList.add("active");
      if (mainContent) mainContent.removeAttribute("data-page");
      chainModalOverlay.classList.add("hidden");
      listDrawerOverlay.classList.add("hidden");
      if (profileDrawerOverlay) profileDrawerOverlay.classList.add("hidden");
      sectionDiscover.classList.add("hidden");
      sectionTrending.classList.remove("hidden");
      sectionEntity.classList.remove("hidden");
      sectionDual.classList.remove("hidden");
      if (updateHash && window.location.hash !== "#home" && window.location.hash !== "") {
        history.pushState(null, "", "#home");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  // Navigation tab click listeners
  navHome.addEventListener("click", (e) => {
    e.preventDefault();
    navigateTo("home");
  });

  navDiscover.addEventListener("click", (e) => {
    e.preventDefault();
    navigateTo("discover");
  });

  navGraph.addEventListener("click", (e) => {
    e.preventDefault();
    navigateTo("graph");
  });

  navMyList.addEventListener("click", (e) => {
    e.preventDefault();
    navigateTo("mylist");
  });

  navFavorites.addEventListener("click", (e) => {
    e.preventDefault();
    navigateTo("favorites");
  });

  // Handle browser Back / Forward navigation
  window.addEventListener("popstate", () => {
    const hash = window.location.hash || "#home";
    navigateTo(hash, false);
  });

  // -------------------------------------------------------------
  // 2. FILTER PILLS (MOVIES, SONGS, BOTH) WITH LIVE FEED FILTERING
  // -------------------------------------------------------------
  filterPills.forEach(pill => {
    pill.addEventListener("click", () => {
      filterPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentMode = pill.dataset.mode || "both";

      // Dynamically load trending feed for the selected mode
      loadTrendingFeed(currentMode);

      // Refresh filter chips for the newly active mode
      renderGenreChips();
      updateFilterUI();

      // If active search results visible or an active seed is selected, re-fetch recommendations
      const q = searchInput.value.trim() || activeSeedTitle;
      if (q && !searchResultsSection.classList.contains("hidden")) {
        executeSearch(q);
      }

      // If Discover Hub is open, refresh with selected mode
      if (!sectionDiscover.classList.contains("hidden")) {
        loadDiscoverContent();
      }
    });
  });

  let cachedTrendingData = null;

  async function loadTrendingFeed(mode = "both") {
    try {
      if (!cachedTrendingData) {
        const res = await fetch("/api/trending");
        cachedTrendingData = await res.json();
      }

      const blockbusters = cachedTrendingData.pan_indian_blockbusters || [];
      const bollywood = cachedTrendingData.modern_bollywood || [];
      const classics = cachedTrendingData.golden_classics || [];
      const songs = cachedTrendingData.viral_hit_songs || [];

      let feedItems = [];
      if (mode === "movies") {
        feedItems = [...blockbusters, ...bollywood, ...classics].map(m => ({ ...m, media_type: "movie" }));
      } else if (mode === "songs") {
        feedItems = songs.map(s => ({ ...s, media_type: "song" }));
      } else {
        // Both: Interleave movies and songs
        const allMovies = [...blockbusters, ...bollywood];
        feedItems = [];
        const maxLen = Math.max(allMovies.length, songs.length);
        for (let i = 0; i < maxLen; i++) {
          if (allMovies[i]) feedItems.push({ ...allMovies[i], media_type: "movie" });
          if (songs[i]) feedItems.push({ ...songs[i], media_type: "song" });
        }
      }

      renderTrendingCarousel(feedItems.slice(0, 16));

      const trendingHeading = document.getElementById("trending-heading");
      if (trendingHeading) {
        if (mode === "movies") trendingHeading.textContent = "Trending Blockbuster Cinema";
        else if (mode === "songs") trendingHeading.textContent = "Trending Hit Melodies & Soundtracks";
        else trendingHeading.textContent = "Trending Right Now";
      }
    } catch (err) {
      console.error("Error loading trending feed:", err);
    }
  }

  function createMediaCard(item) {
    const isMovie = (item.media_type || "movie").toLowerCase() === "movie";
    const title = item.title || "Untitled";
    const poster = isMovie 
      ? (item.poster_url || "/static/images/poster_rockstar.jpg")
      : (item.thumbnail_url || "/static/images/poster_kesariya.jpg");
    const sub = isMovie ? (item.genres || "Cinema") : (item.singer || "Track");
    const meta = isMovie ? (item.director || item.release_year || "Blockbuster") : (item.movie_title || item.composer || "Hit Single");
    const isSaved = isItemSaved(title);

    const trailerUrl = isMovie 
      ? `https://www.youtube.com/results?search_query=${encodeURIComponent(title + " official trailer")}`
      : `https://www.youtube.com/results?search_query=${encodeURIComponent(title + " official video song")}`;

    const ratingInfo = getItemRating(title, isMovie ? "movie" : "song");

    const card = document.createElement("div");
    card.className = "poster-card";
    card.dataset.title = title;
    card.dataset.type = isMovie ? "movie" : "song";

    let barsHTML = "";
    for (let i = 1; i <= 5; i++) {
      barsHTML += `<span class="meter-bar-segment ${i <= ratingInfo.level ? 'active' : ''}"></span>`;
    }

    card.innerHTML = `
      <div class="poster-media">
        <img src="${poster}" alt="${escapeHtml(title)}" onerror="this.src='/static/images/poster_dil_chahta_hai.jpg'" loading="lazy" />
        <button type="button" class="card-rating-meter" title="Audience Vibe Rating: ${escapeHtml(ratingInfo.label)} · Click to rate" data-title="${escapeHtml(title)}" data-type="${isMovie ? 'movie' : 'song'}">
          <span class="meter-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
          </span>
          <span class="rating-dot" style="background-color: ${ratingInfo.color}; box-shadow: 0 0 7px ${ratingInfo.color};"></span>
          <span class="rating-label">${escapeHtml(ratingInfo.label)}</span>
          <span class="meter-bars">${barsHTML}</span>
        </button>
        <button class="card-save-btn ${isSaved ? 'saved' : ''}" title="Save to list" data-title="${escapeHtml(title)}" data-type="${isMovie ? 'movie' : 'song'}">
          <svg viewBox="0 0 24 24" fill="${isSaved ? '#fbbf24' : 'none'}" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>
        </button>
        <span class="type-pill ${isMovie ? 'pill-blue' : 'pill-purple'}">${isMovie ? 'Movie' : 'Music'}</span>
      </div>
      <div class="poster-body">
        <h3 class="card-title">${escapeHtml(title)}</h3>
        <p class="card-genre">${escapeHtml(sub)}</p>
        <div class="card-tags-row">
          <span class="tag-chip">${escapeHtml(meta)}</span>
        </div>
        <div class="card-action-row">
          <a href="${trailerUrl}" target="_blank" rel="noopener noreferrer" class="card-btn-trailer" title="${isMovie ? 'Watch official trailer on YouTube' : 'Watch official video song on YouTube'}">
            ${isMovie ? '🎬 Trailer' : '▶ Video Song'}
          </a>
          <button type="button" class="card-btn-watch" title="${isMovie ? 'Where to watch (JioHotstar, Netflix, YouTube)' : 'Where to stream (YouTube, Spotify, etc.)'}">
            ${isMovie ? '⚡ Watch' : '⚡ Stream'}
          </button>
        </div>
      </div>
    `;

    // 1. Click card body -> Open Watch & Streaming Options Modal
    card.addEventListener("click", () => {
      openWatchModal(title, isMovie ? "movie" : "song");
    });

    // 2. Click rating meter -> Open Quick Rating Popover
    const ratingBtn = card.querySelector(".card-rating-meter");
    if (ratingBtn) {
      ratingBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        openRatingPopover(ratingBtn, title, isMovie ? "movie" : "song");
      });
    }

    // 3. Click trailer link -> don't bubble to card modal
    const trailerBtn = card.querySelector(".card-btn-trailer");
    if (trailerBtn) {
      trailerBtn.addEventListener("click", (e) => {
        e.stopPropagation();
      });
    }

    // 4. Click watch button -> Open Watch Modal
    const watchBtn = card.querySelector(".card-btn-watch");
    if (watchBtn) {
      watchBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        openWatchModal(title, isMovie ? "movie" : "song");
      });
    }

    // 5. Click save button
    const saveBtn = card.querySelector(".card-save-btn");
    if (saveBtn) {
      saveBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        toggleSaveItem(title, isMovie ? "movie" : "song", sub, saveBtn);
      });
    }

    attachFluid3DTilt(card);
    return card;
  }

  function renderTrendingCarousel(items) {
    if (!trendingTrack) return;
    trendingTrack.innerHTML = "";

    if (!items || !items.length) {
      trendingTrack.innerHTML = `<div style="padding: 24px; color: #94a3b8;">No trending titles found for this category.</div>`;
      return;
    }

    items.forEach(item => {
      const card = createMediaCard(item);
      trendingTrack.appendChild(card);
    });
  }

  function selectAndRecommend(title, type = "movie") {
    if (!title) return;
    activeSeedTitle = title;
    activeSeedType = type;

    // Highlight selected card visually
    document.querySelectorAll(".poster-card").forEach(c => {
      if (c.dataset.title === title) {
        c.classList.add("card-selected");
      } else {
        c.classList.remove("card-selected");
      }
    });

    searchInput.value = title;
    searchClearBtn.classList.remove("hidden");
    showAudioToast(title, type === "movie" ? "Selected Movie · Finding AI Matches" : "Selected Song · Finding AI Matches");
    executeSearch(title);
  }

  // -------------------------------------------------------------
  // 3. SEARCH & AUTOCOMPLETE
  // -------------------------------------------------------------
  searchInput.addEventListener("input", (e) => {
    const val = e.target.value;
    if (val.trim()) {
      searchClearBtn.classList.remove("hidden");
    } else {
      searchClearBtn.classList.add("hidden");
      hideAutocomplete();
    }

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      fetchAutocomplete(val);
    }, 200);
  });

  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const val = searchInput.value.trim();
      if (val) {
        hideAutocomplete();
        executeSearch(val);
      }
    } else if (e.key === "Escape") {
      hideAutocomplete();
    }
  });

  searchClearBtn.addEventListener("click", () => {
    searchInput.value = "";
    searchClearBtn.classList.add("hidden");
    hideAutocomplete();
    searchInput.focus();
  });

  // Voice Search (Microphone)
  if (searchMicBtn) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-IN";

      searchMicBtn.addEventListener("click", () => {
        searchMicBtn.classList.add("listening");
        recognition.start();
      });

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        searchInput.value = transcript;
        searchClearBtn.classList.remove("hidden");
        searchMicBtn.classList.remove("listening");
        executeSearch(transcript);
      };

      recognition.onerror = () => {
        searchMicBtn.classList.remove("listening");
      };

      recognition.onend = () => {
        searchMicBtn.classList.remove("listening");
      };
    } else {
      searchMicBtn.addEventListener("click", () => {
        searchInput.focus();
      });
    }
  }

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-bar-wrap")) {
      hideAutocomplete();
    }
  });

  async function fetchAutocomplete(query) {
    if (!query || query.trim().length < 2) {
      hideAutocomplete();
      return;
    }

    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(query.trim())}&limit=6`);
      const data = await res.json();
      const results = data.results || [];
      renderAutocomplete(results);
    } catch (err) {
      console.error("Autocomplete fetch error:", err);
    }
  }

  function renderAutocomplete(results) {
    if (!results.length) {
      hideAutocomplete();
      return;
    }

    autocompleteItems.innerHTML = "";
    results.forEach(item => {
      const row = document.createElement("div");
      row.className = "autocomplete-row";

      const t = (item.type || "").toLowerCase();
      const typeClass = t === "movie" ? "auto-type-movie" : (t === "song" ? "auto-type-song" : "auto-type-artist");
      const sub = item.badge || item.subtext || item.relation || item.era || "";

      row.innerHTML = `
        <span class="auto-type-badge ${typeClass}">${escapeHtml(item.type)}</span>
        <span class="auto-title">${escapeHtml(item.title)}</span>
        <span class="auto-sub">${escapeHtml(sub)}</span>
      `;

      row.addEventListener("click", () => {
        hideAutocomplete();
        selectAndRecommend(item.title, t === "song" ? "song" : "movie");
      });

      autocompleteItems.appendChild(row);
    });

    autocompletePanel.classList.remove("hidden");
  }

  function hideAutocomplete() {
    autocompletePanel.classList.add("hidden");
  }

  // -------------------------------------------------------------
  // MULTI-GENRE & YEAR FILTER INTELLIGENCE SYSTEM
  // -------------------------------------------------------------
  async function initFilterIntelligence() {
    try {
      const res = await fetch("/api/filter-options");
      cachedFilterOptions = await res.json();
      populateYearOptions();
      renderGenreChips();
      renderYearPresets();
    } catch (err) {
      console.error("Error loading filter options:", err);
    }
  }

  function populateYearOptions() {
    if (!cachedFilterOptions || !exactYearSelect) return;
    exactYearSelect.innerHTML = `<option value="All">All Years</option>`;
    (cachedFilterOptions.years || []).forEach(y => {
      const opt = document.createElement("option");
      opt.value = y;
      opt.textContent = y;
      exactYearSelect.appendChild(opt);
    });
  }

  function renderGenreChips() {
    if (!genreChipsContainer || !cachedFilterOptions) return;
    genreChipsContainer.innerHTML = "";

    let genresList = [];
    if (currentMode === "movies") {
      genresList = cachedFilterOptions.movie_genres || [];
      if (filterModeIndicator) filterModeIndicator.textContent = "🎥 Movies Mode";
      if (genreHintText) genreHintText.textContent = "Select multiple cinema genres (e.g., Action + Thriller + Sci-Fi)";
    } else if (currentMode === "songs") {
      genresList = cachedFilterOptions.music_genres || [];
      if (filterModeIndicator) filterModeIndicator.textContent = "🎵 Music Mode";
      if (genreHintText) genreHintText.textContent = "Select multiple music genres (e.g., Indian Indie + Soulful + Pop)";
    } else {
      genresList = cachedFilterOptions.both_genres || [];
      if (filterModeIndicator) filterModeIndicator.textContent = "✨ Both Mode";
      if (genreHintText) genreHintText.textContent = "Select cross-domain genres to find matching cinema and songs";
    }

    genresList.forEach(genre => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = `genre-chip-toggle ${selectedMultiGenres.has(genre) ? "active" : ""}`;
      chip.dataset.genre = genre;
      chip.innerHTML = `
        <span class="chip-check-icon">${selectedMultiGenres.has(genre) ? "✓" : "+"}</span>
        <span>${escapeHtml(genre)}</span>
      `;

      chip.addEventListener("click", () => {
        if (selectedMultiGenres.has(genre)) {
          selectedMultiGenres.delete(genre);
          chip.classList.remove("active");
          chip.querySelector(".chip-check-icon").textContent = "+";
        } else {
          selectedMultiGenres.add(genre);
          chip.classList.add("active");
          chip.querySelector(".chip-check-icon").textContent = "✓";
        }
        updateFilterUI();
      });

      genreChipsContainer.appendChild(chip);
    });
  }

  function renderYearPresets() {
    if (!yearPresetsChips || !cachedFilterOptions) return;
    yearPresetsChips.innerHTML = "";

    const presets = cachedFilterOptions.year_presets || [];
    presets.forEach(p => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = `year-preset-chip ${selectedYearFilter === p.value ? "active" : ""}`;
      chip.dataset.year = p.value;
      chip.textContent = p.label;

      chip.addEventListener("click", () => {
        selectedYearFilter = p.value;
        if (exactYearSelect) exactYearSelect.value = "All";
        document.querySelectorAll(".year-preset-chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        updateFilterUI();
      });

      yearPresetsChips.appendChild(chip);
    });
  }

  if (exactYearSelect) {
    exactYearSelect.addEventListener("change", (e) => {
      const val = e.target.value;
      selectedYearFilter = val;
      document.querySelectorAll(".year-preset-chip").forEach(c => {
        c.classList.toggle("active", c.dataset.year === val);
      });
      updateFilterUI();
    });
  }

  function updateFilterUI() {
    const totalFilters = selectedMultiGenres.size + (selectedYearFilter !== "All" ? 1 : 0);
    
    // Update badge on Search Filters toggle button
    if (activeFiltersCount) {
      if (totalFilters > 0) {
        activeFiltersCount.textContent = totalFilters;
        activeFiltersCount.classList.remove("hidden");
        searchFiltersToggleBtn.classList.add("active");
      } else {
        activeFiltersCount.textContent = "0";
        activeFiltersCount.classList.add("hidden");
        if (searchFilterPanel && searchFilterPanel.classList.contains("hidden")) {
          searchFiltersToggleBtn.classList.remove("active");
        }
      }
    }

    // Render Active Tags Summary inside Filter Panel Footer
    if (activeTagsSummary) {
      activeTagsSummary.innerHTML = "";
      if (totalFilters === 0) {
        activeTagsSummary.innerHTML = `<span class="no-tags-hint">No filters selected. Click any genre chips or years above.</span>`;
      } else {
        selectedMultiGenres.forEach(genre => {
          const tag = document.createElement("span");
          tag.className = "active-chip-tag";
          tag.innerHTML = `<span>${escapeHtml(genre)}</span><span class="tag-remove">✕</span>`;
          tag.addEventListener("click", () => {
            selectedMultiGenres.delete(genre);
            renderGenreChips();
            updateFilterUI();
          });
          activeTagsSummary.appendChild(tag);
        });

        if (selectedYearFilter !== "All") {
          const yTag = document.createElement("span");
          yTag.className = "active-chip-tag";
          yTag.innerHTML = `<span>📅 ${escapeHtml(selectedYearFilter)}</span><span class="tag-remove">✕</span>`;
          yTag.addEventListener("click", () => {
            selectedYearFilter = "All";
            if (exactYearSelect) exactYearSelect.value = "All";
            renderYearPresets();
            updateFilterUI();
          });
          activeTagsSummary.appendChild(yTag);
        }
      }
    }

    // Render Active Filters Bar under Search Bar
    if (activeFiltersBar && activeFilterChipsList) {
      activeFilterChipsList.innerHTML = "";
      if (totalFilters === 0) {
        activeFiltersBar.classList.add("hidden");
      } else {
        activeFiltersBar.classList.remove("hidden");
        selectedMultiGenres.forEach(genre => {
          const tag = document.createElement("span");
          tag.className = "active-chip-tag";
          tag.innerHTML = `<span>${escapeHtml(genre)}</span><span class="tag-remove">✕</span>`;
          tag.addEventListener("click", () => {
            selectedMultiGenres.delete(genre);
            renderGenreChips();
            updateFilterUI();
            executeSearch(searchInput.value.trim());
          });
          activeFilterChipsList.appendChild(tag);
        });

        if (selectedYearFilter !== "All") {
          const yTag = document.createElement("span");
          yTag.className = "active-chip-tag";
          yTag.innerHTML = `<span>📅 ${escapeHtml(selectedYearFilter)}</span><span class="tag-remove">✕</span>`;
          yTag.addEventListener("click", () => {
            selectedYearFilter = "All";
            if (exactYearSelect) exactYearSelect.value = "All";
            renderYearPresets();
            updateFilterUI();
            executeSearch(searchInput.value.trim());
          });
          activeFilterChipsList.appendChild(yTag);
        }
      }
    }
  }

  function resetAllFilters() {
    selectedMultiGenres.clear();
    selectedYearFilter = "All";
    if (exactYearSelect) exactYearSelect.value = "All";
    renderGenreChips();
    renderYearPresets();
    updateFilterUI();
  }

  if (filterResetBtn) filterResetBtn.addEventListener("click", resetAllFilters);
  if (clearAllFiltersBtn) {
    clearAllFiltersBtn.addEventListener("click", () => {
      resetAllFilters();
      const q = searchInput.value.trim();
      if (q) {
        executeSearch(q);
      } else {
        searchResultsSection.classList.add("hidden");
      }
    });
  }

  if (searchFiltersToggleBtn) {
    searchFiltersToggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isHidden = searchFilterPanel.classList.contains("hidden");
      if (isHidden) {
        searchFilterPanel.classList.remove("hidden");
        searchFiltersToggleBtn.classList.add("active");
        renderGenreChips();
      } else {
        searchFilterPanel.classList.add("hidden");
        if (selectedMultiGenres.size === 0 && selectedYearFilter === "All") {
          searchFiltersToggleBtn.classList.remove("active");
        }
      }
    });
  }

  if (filterCloseBtn) {
    filterCloseBtn.addEventListener("click", () => {
      searchFilterPanel.classList.add("hidden");
      if (selectedMultiGenres.size === 0 && selectedYearFilter === "All") {
        searchFiltersToggleBtn.classList.remove("active");
      }
    });
  }

  if (applyFiltersBtn) {
    applyFiltersBtn.addEventListener("click", () => {
      searchFilterPanel.classList.add("hidden");
      const q = searchInput.value.trim();
      executeSearch(q);
    });
  }

  // Recommendation Execution
  async function executeSearch(query) {
    const hasFilters = selectedMultiGenres.size > 0 || selectedYearFilter !== "All";
    if (!query && !hasFilters) return;

    const genresParam = Array.from(selectedMultiGenres).join(",");
    const yearParam = selectedYearFilter;

    let displayQuery = query;
    if (!displayQuery && hasFilters) {
      displayQuery = Array.from(selectedMultiGenres).join(" + ") + (selectedYearFilter !== "All" ? ` (${selectedYearFilter})` : "");
    }

    activeSeedTitle = displayQuery;
    activeSeedType = currentMode === "songs" ? "song" : "movie";

    queryHighlight.textContent = displayQuery;
    resultsDetail.textContent = `AI engine analyzing combinatorial weights: ${genresParam || "All Genres"} · Year: ${yearParam}...`;
    searchResultsSection.classList.remove("hidden");
    sectionDiscover.classList.add("hidden");
    searchResultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    try {
      const url = `/api/recommend?query=${encodeURIComponent(query || "")}&genres=${encodeURIComponent(genresParam)}&year=${encodeURIComponent(yearParam)}&mode=${currentMode}&top_n=12`;
      const res = await fetch(url);
      const data = await res.json();
      renderSearchResults(data);
    } catch (err) {
      console.error("Search error:", err);
      resultsDetail.textContent = "Unable to fetch recommendations. Please try another query.";
    }
  }

  function renderSearchResults(data) {
    const seed = data.seed || data.identified;
    if (seed) {
      activeSeedTitle = seed.title || activeSeedTitle;
      activeSeedType = (seed.type || "movie").toLowerCase();
      resultsBadge.textContent = `${(seed.type || "Identified").toUpperCase()}: ${seed.title || activeSeedTitle}`;
      resultsDetail.textContent = seed.explanation || "Synergistic cross-domain entertainment discovered.";
      queryHighlight.textContent = seed.title || activeSeedTitle;
    }

    const movies = data.movies || [];
    const songs = data.songs || [];
    const exactMatch = data.exact_match || (movies.length > 0 && movies[0].is_searched_title ? movies[0] : null) || (songs.length > 0 && songs[0].is_searched_title ? songs[0] : null);

    // 1. Render Featured Exact Match Spotlight (Spotlight for the searched title)
    const spotlightCard = document.getElementById("search-spotlight-card");
    if (spotlightCard) {
      if (exactMatch) {
        const isMovie = (exactMatch.media_type || (exactMatch.actors ? "movie" : "song")).toLowerCase() === "movie";
        const title = exactMatch.title || activeSeedTitle;
        const year = exactMatch.release_year || "";
        const genres = exactMatch.genres || exactMatch.mood || "Featured Title";
        const directorOrSinger = isMovie ? (exactMatch.director ? `Directed by ${exactMatch.director}` : "") : (exactMatch.singer ? `Vocals by ${exactMatch.singer}` : "");
        const stars = isMovie && exactMatch.actors ? `Starring ${exactMatch.actors}` : "";
        const overview = exactMatch.overview || exactMatch.lyrics_snippet || `Original ${genres} ${isMovie ? 'feature film' : 'track'} released in ${year}.`;
        const posterUrl = exactMatch.poster_url || exactMatch.thumbnail_url || (isMovie ? "/static/images/poster_znmd.jpg" : "/static/images/poster_rockstar.jpg");

        spotlightCard.innerHTML = `
          <div class="spotlight-poster-wrap">
            <img src="${posterUrl}" alt="${escapeHtml(title)}" class="spotlight-poster-img" onerror="this.src='/static/images/poster_znmd.jpg'" />
          </div>
          <div class="spotlight-body">
            <div class="spotlight-badge-row">
              <span class="spotlight-badge-target">🎯 TARGET MATCH</span>
              <span class="spotlight-badge-meta">${isMovie ? '🎬 Movie' : '🎵 Song'}</span>
              ${year ? `<span class="spotlight-badge-meta">${year}</span>` : ''}
              <span class="spotlight-badge-meta">${escapeHtml(genres)}</span>
            </div>
            <h2 class="spotlight-title">${escapeHtml(title)}</h2>
            <div class="spotlight-credits">
              ${directorOrSinger ? `<strong>${escapeHtml(directorOrSinger)}</strong>` : ''}
              ${stars ? ` &bull; <span>${escapeHtml(stars)}</span>` : ''}
            </div>
            <p class="spotlight-overview">${escapeHtml(overview)}</p>
            <div class="spotlight-actions-row">
              <button type="button" class="spotlight-btn-watch" id="spotlight-watch-btn">
                <span>🎬 Watch Options</span>
              </button>
              <button type="button" class="spotlight-btn-chain" id="spotlight-chain-btn">
                <span>🕸️ Relational Chain</span>
              </button>
              <button type="button" class="spotlight-btn-save" id="spotlight-save-btn">
                <span>📑 Save to List</span>
              </button>
            </div>
          </div>
        `;
        spotlightCard.classList.remove("hidden");

        // Event listeners for spotlight card
        const spWatchBtn = spotlightCard.querySelector("#spotlight-watch-btn");
        if (spWatchBtn) {
          spWatchBtn.addEventListener("click", () => {
            openWatchModal(title, isMovie ? "movie" : "song");
          });
        }
        const spChainBtn = spotlightCard.querySelector("#spotlight-chain-btn");
        if (spChainBtn) {
          spChainBtn.addEventListener("click", () => {
            openChainModal(isMovie ? "movie" : "song", title);
          });
        }
        const spSaveBtn = spotlightCard.querySelector("#spotlight-save-btn");
        if (spSaveBtn) {
          spSaveBtn.addEventListener("click", () => {
            toggleSaveItem(title, isMovie ? "movie" : "song", genres, spSaveBtn);
          });
        }
      } else {
        spotlightCard.classList.add("hidden");
      }
    }

    // 2. Render Similar Movies Column
    const moviesGrid = document.getElementById("search-movies-grid");
    const moviesCol = document.getElementById("search-movies-column");
    const badgeMoviesCount = document.getElementById("badge-movies-count");
    const countMovies = document.getElementById("count-movies");

    if (moviesGrid && moviesCol) {
      moviesGrid.innerHTML = "";
      if (movies.length > 0) {
        movies.forEach(m => {
          const card = createMediaCard({ ...m, media_type: 'movie' });
          moviesGrid.appendChild(card);
        });
        moviesCol.classList.remove("hidden");
        if (badgeMoviesCount) badgeMoviesCount.textContent = movies.length;
        if (countMovies) countMovies.textContent = movies.length;
      } else {
        moviesCol.classList.add("hidden");
        if (countMovies) countMovies.textContent = "0";
      }
    }

    // 3. Render Related Songs Column
    const songsGrid = document.getElementById("search-songs-grid");
    const songsCol = document.getElementById("search-songs-column");
    const badgeSongsCount = document.getElementById("badge-songs-count");
    const countSongs = document.getElementById("count-songs");

    if (songsGrid && songsCol) {
      songsGrid.innerHTML = "";
      if (songs.length > 0) {
        songs.forEach(s => {
          const card = createMediaCard({ ...s, media_type: 'song' });
          songsGrid.appendChild(card);
        });
        songsCol.classList.remove("hidden");
        if (badgeSongsCount) badgeSongsCount.textContent = songs.length;
        if (countSongs) countSongs.textContent = songs.length;
      } else {
        songsCol.classList.add("hidden");
        if (countSongs) countSongs.textContent = "0";
      }
    }

    // 4. Update Subnav counts
    const countAll = document.getElementById("count-all");
    const totalCount = movies.length + songs.length;
    if (countAll) countAll.textContent = totalCount;

    // Subnav Tab Filtering
    const tabAll = document.getElementById("tab-results-all");
    const tabMovies = document.getElementById("tab-results-movies");
    const tabSongs = document.getElementById("tab-results-songs");

    function setResultsActiveTab(activeTab) {
      [tabAll, tabMovies, tabSongs].forEach(t => t && t.classList.remove("active"));
      if (activeTab) activeTab.classList.add("active");
    }

    if (tabAll) {
      tabAll.onclick = () => {
        setResultsActiveTab(tabAll);
        if (moviesCol && movies.length) moviesCol.classList.remove("hidden");
        if (songsCol && songs.length) songsCol.classList.remove("hidden");
      };
    }
    if (tabMovies) {
      tabMovies.onclick = () => {
        setResultsActiveTab(tabMovies);
        if (moviesCol && movies.length) moviesCol.classList.remove("hidden");
        if (songsCol) songsCol.classList.add("hidden");
      };
    }
    if (tabSongs) {
      tabSongs.onclick = () => {
        setResultsActiveTab(tabSongs);
        if (moviesCol) moviesCol.classList.add("hidden");
        if (songsCol && songs.length) songsCol.classList.remove("hidden");
      };
    }

    // Reset to "All" tab by default
    if (tabAll) setResultsActiveTab(tabAll);

    // Empty state fallback
    if (totalCount === 0) {
      searchResultsGrid.classList.remove("hidden");
      searchResultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 36px; text-align: center; color: #94a3b8; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px dashed rgba(255,255,255,0.08);">
        <p style="font-size: 16px; color: #e2e8f0; margin-bottom: 6px;">No exact matches found for "${escapeHtml(activeSeedTitle)}"</p>
        <p style="font-size: 12px; color: #64748b;">Try searching another Bollywood title, singer, director, or use the multi-genre and year filters!</p>
      </div>`;
    } else {
      searchResultsGrid.classList.add("hidden");
    }
  }

  closeResultsBtn.addEventListener("click", () => {
    searchResultsSection.classList.add("hidden");
  });

  exploreChainTopBtn.addEventListener("click", () => {
    openChainModal(activeSeedType, activeSeedTitle);
  });

  // -------------------------------------------------------------
  // 4. DISCOVER HUB LOGIC
  // -------------------------------------------------------------
  eraChips.forEach(chip => {
    chip.addEventListener("click", () => {
      eraChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      selectedEra = chip.dataset.era;
      loadDiscoverContent();
    });
  });

  genreChips.forEach(chip => {
    chip.addEventListener("click", () => {
      genreChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      selectedGenre = chip.dataset.genre;
      loadDiscoverContent();
    });
  });

  async function loadDiscoverContent() {
    discoverResultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 24px; text-align: center; color: #94a3b8;">Loading curated titles for ${selectedEra} &bull; ${selectedGenre}...</div>`;

    try {
      const q = selectedGenre !== "All" ? selectedGenre : (currentMode === "songs" ? "Melodies" : "Blockbuster");
      const res = await fetch(`/api/recommend?query=${encodeURIComponent(q)}&era=${encodeURIComponent(selectedEra)}&genre=${encodeURIComponent(selectedGenre)}&mode=${currentMode}&top_n=12`);
      const data = await res.json();
      
      const movies = data.movies || [];
      const songs = data.songs || [];
      const items = [...movies.map(m => ({ ...m, media_type: 'movie' })), ...songs.map(s => ({ ...s, media_type: 'song' }))];

      discoverResultsGrid.innerHTML = "";
      if (!items.length) {
        discoverResultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 24px; text-align: center; color: #94a3b8;">No titles found for this filter combination. Try another era or genre!</div>`;
        return;
      }

      items.forEach(item => {
        const card = createMediaCard(item);
        discoverResultsGrid.appendChild(card);
      });
    } catch (err) {
      console.error("Discover load error:", err);
      discoverResultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 20px; text-align: center; color: #ef4444;">Could not load discover content.</div>`;
    }
  }

  // -------------------------------------------------------------
  // 5. MY LIST & FAVORITES DRAWER & LOCALSTORAGE
  // -------------------------------------------------------------
  function isItemSaved(title) {
    return myList.some(item => item.title === title) || myFavorites.some(item => item.title === title);
  }

  function toggleSaveItem(title, type, sub, btnEl) {
    const targetArray = (currentDrawerType === "favorites") ? myFavorites : myList;
    const storageKey = (currentDrawerType === "favorites") ? "cinevibe_favorites" : "cinevibe_my_list";
    const legacyKey = (currentDrawerType === "favorites") ? "velora_favorites" : "velora_my_list";

    const idx = targetArray.findIndex(item => item.title === title);
    if (idx >= 0) {
      targetArray.splice(idx, 1);
      if (btnEl) {
        btnEl.classList.remove("saved");
        const svg = btnEl.querySelector("svg");
        if (svg) svg.setAttribute("fill", "none");
      }
      showAudioToast(`Removed "${title}"`, "List Updated");
    } else {
      targetArray.push({ title, type, sub, date: new Date().toLocaleDateString() });
      if (btnEl) {
        btnEl.classList.add("saved");
        const svg = btnEl.querySelector("svg");
        if (svg) svg.setAttribute("fill", "#fbbf24");
      }
      showAudioToast(`Saved "${title}"`, "Added to Collection");
    }

    localStorage.setItem(storageKey, JSON.stringify(targetArray));
    localStorage.setItem(legacyKey, JSON.stringify(targetArray));
    updateBadges();

    // If drawer currently open, refresh its list
    if (!listDrawerOverlay.classList.contains("hidden")) {
      renderDrawerList();
    }
  }

  function updateBadges() {
    if (myList.length > 0) {
      badgeMyList.textContent = myList.length;
      badgeMyList.classList.remove("hidden");
    } else {
      badgeMyList.classList.add("hidden");
    }

    if (myFavorites.length > 0) {
      badgeFavorites.textContent = myFavorites.length;
      badgeFavorites.classList.remove("hidden");
    } else {
      badgeFavorites.classList.add("hidden");
    }
  }

  function openDrawer(type) {
    currentDrawerType = type;
    drawerBadge.textContent = type === "favorites" ? "Starred Favourites" : "Saved Collection";
    drawerTitle.textContent = type === "favorites" ? "Favorites" : "My List";
    renderDrawerList();
    listDrawerOverlay.classList.remove("hidden");
  }

  function renderDrawerList() {
    const list = currentDrawerType === "favorites" ? myFavorites : myList;
    drawerItemsList.innerHTML = "";

    if (!list.length) {
      drawerItemsList.innerHTML = `
        <div class="empty-drawer-msg">
          <p>Your ${currentDrawerType === "favorites" ? "favorites" : "list"} is empty.</p>
          <p style="font-size: 11px; margin-top: 6px; color: #64748b;">Click the bookmark icon on any card to save it here for instant access!</p>
        </div>
      `;
      return;
    }

    list.forEach(item => {
      const row = document.createElement("div");
      row.className = "drawer-row-item";
      row.innerHTML = `
        <div class="drawer-row-left">
          <span class="drawer-row-icon">${item.type === 'movie' ? '🎬' : '🎵'}</span>
          <div class="drawer-row-info">
            <span class="drawer-row-title">${escapeHtml(item.title)}</span>
            <span class="drawer-row-sub">${escapeHtml(item.sub || item.type)} &bull; ${item.date || ''}</span>
          </div>
        </div>
        <div class="drawer-row-actions">
          <button class="drawer-item-play-btn" data-title="${escapeHtml(item.title)}">Explore</button>
          <button class="drawer-item-remove-btn" title="Remove" data-title="${escapeHtml(item.title)}">✕</button>
        </div>
      `;

      row.querySelector(".drawer-item-play-btn").addEventListener("click", () => {
        listDrawerOverlay.classList.add("hidden");
        navigateTo("home", false);
        searchInput.value = item.title;
        searchClearBtn.classList.remove("hidden");
        executeSearch(item.title);
      });

      row.querySelector(".drawer-item-remove-btn").addEventListener("click", () => {
        toggleSaveItem(item.title, item.type, item.sub, null);
      });

      drawerItemsList.appendChild(row);
    });
  }

  drawerCloseBtn.addEventListener("click", () => {
    listDrawerOverlay.classList.add("hidden");
    navigateTo("home");
  });

  listDrawerOverlay.addEventListener("click", (e) => {
    if (e.target === listDrawerOverlay) {
      listDrawerOverlay.classList.add("hidden");
    }
  });

  drawerClearBtn.addEventListener("click", () => {
    if (confirm(`Clear all items from ${currentDrawerType === "favorites" ? "Favorites" : "My List"}?`)) {
      if (currentDrawerType === "favorites") {
        myFavorites = [];
        localStorage.setItem("velora_favorites", "[]");
        localStorage.setItem("cinevibe_favorites", "[]");
      } else {
        myList = [];
        localStorage.setItem("velora_my_list", "[]");
        localStorage.setItem("cinevibe_my_list", "[]");
      }
      updateBadges();
      renderDrawerList();
      // Remove saved class from cards on page
      document.querySelectorAll(".card-save-btn").forEach(btn => {
        btn.classList.remove("saved");
        const svg = btn.querySelector("svg");
        if (svg) svg.setAttribute("fill", "none");
      });
    }
  });

  // -------------------------------------------------------------
  // 6. INITIAL CARD LISTENERS & CAROUSELS
  // -------------------------------------------------------------
  // Card save buttons on initial HTML
  document.querySelectorAll(".card-save-btn").forEach(btn => {
    const title = btn.dataset.title;
    const type = btn.dataset.type || "movie";
    if (isItemSaved(title)) {
      btn.classList.add("saved");
      const svg = btn.querySelector("svg");
      if (svg) svg.setAttribute("fill", "#fbbf24");
    }

    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSaveItem(title, type, "", btn);
    });
  });

  // Initial card clicks & Polaroid clicks
  document.querySelectorAll(".poster-card").forEach(card => {
    const title = card.dataset.title;
    const type = card.dataset.type || "movie";
    if (title) {
      card.addEventListener("click", () => {
        selectAndRecommend(title, type);
      });

      const streamBtn = card.querySelector(".stream-badge-pill");
      if (streamBtn) {
        streamBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          e.preventDefault();
          openWatchModal(title, type);
        });
      }
    }
  });

  // Hero Polaroid Card Clicks -> Select & Recommend
  document.querySelectorAll(".polaroid-card").forEach((card, idx) => {
    const titles = ["Zindagi Na Milegi Dobara", "Dil Chahta Hai", "Rockstar"];
    const t = titles[idx] || "Rockstar";
    card.style.cursor = "pointer";
    card.setAttribute("title", `Select & Explore "${t}"`);
    card.addEventListener("click", (e) => {
      e.stopPropagation();
      selectAndRecommend(t, "movie");
    });
  });

  // Initial load of trending feed matching active mode
  loadTrendingFeed(currentMode);

  const viewAllTrending = document.getElementById("view-all-trending");
  if (viewAllTrending) {
    viewAllTrending.addEventListener("click", (e) => {
      e.preventDefault();
      setActiveNavTab(navDiscover);
      sectionDiscover.classList.remove("hidden");
      sectionDiscover.scrollIntoView({ behavior: "smooth", block: "start" });
      loadDiscoverContent();
    });
  }

  // Carousel arrow controls
  if (btnTrendingNext && trendingTrack) {
    btnTrendingNext.addEventListener("click", () => {
      trendingTrack.scrollBy({ left: 320, behavior: "smooth" });
    });
  }

  if (btnEntityNext && entityTrack) {
    btnEntityNext.addEventListener("click", () => {
      entityTrack.scrollBy({ left: 300, behavior: "smooth" });
    });
  }

  // Entity item clicks
  document.querySelectorAll(".entity-item").forEach(item => {
    item.addEventListener("click", () => {
      const type = item.dataset.type || "actor";
      const name = item.dataset.name;
      if (name) {
        openChainModal(type, name);
      }
    });
  });

  // Dual Cards Explore button clicks
  document.querySelectorAll(".dual-explore-btn, .dual-card").forEach(el => {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      const title = el.dataset.query || el.dataset.title;
      if (title) {
        searchInput.value = title;
        searchClearBtn.classList.remove("hidden");
        executeSearch(title);
      }
    });
  });

  // Recent Searches clicks
  document.querySelectorAll(".recent-item").forEach(item => {
    item.addEventListener("click", () => {
      const q = item.dataset.query;
      if (q) {
        searchInput.value = q;
        searchClearBtn.classList.remove("hidden");
        executeSearch(q);
      }
    });
  });

  // Quick Explore Buttons
  document.querySelectorAll(".quick-tile").forEach(tile => {
    tile.addEventListener("click", () => {
      const action = tile.dataset.action;
      if (action === "top_movies") {
        currentMode = "movies";
        updateActiveFilterPill("movies");
        searchInput.value = "RRR";
        executeSearch("RRR");
      } else if (action === "top_songs") {
        currentMode = "songs";
        updateActiveFilterPill("songs");
        searchInput.value = "Kesariya";
        executeSearch("Kesariya");
      } else if (action === "trending_artists") {
        openChainModal("actor", "Shah Rukh Khan");
      } else if (action === "latest_releases") {
        searchInput.value = "Stree 2";
        executeSearch("Stree 2");
      }
    });
  });

  function updateActiveFilterPill(mode) {
    filterPills.forEach(p => {
      if (p.dataset.mode === mode) p.classList.add("active");
      else p.classList.remove("active");
    });
  }

  // Floating Back to Top Button
  if (floatingTopBtn) {
    floatingTopBtn.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
      const mainContent = document.getElementById("main-content");
      if (mainContent) mainContent.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // -------------------------------------------------------------
  // 7. KNOWLEDGE GRAPH MODAL & QUICK ENTITY PICKER
  // -------------------------------------------------------------
  async function openChainModal(type, name) {
    if (!name) return;
    modalBadge.textContent = type.toUpperCase();
    modalName.textContent = name;
    modalStats.innerHTML = `<div class="modal-stat-pill"><div class="modal-stat-val">...</div><div class="modal-stat-lbl">Crawling graph</div></div>`;
    modalCollabsList.innerHTML = "";
    modalMoviesList.innerHTML = "";
    modalSongsList.innerHTML = "";
    chainModalOverlay.classList.remove("hidden");

    try {
      const res = await fetch(`/api/chain?type=${encodeURIComponent(type)}&name=${encodeURIComponent(name)}`);
      const data = await res.json();
      renderChainModal(data);
    } catch (err) {
      console.error("Chain API error:", err);
    }
  }

  function renderChainModal(data) {
    const movies = data.movies || [];
    const songs = data.songs || [];
    const collabs = data.collaborations || [];

    modalStats.innerHTML = `
      <div class="modal-stat-pill">
        <div class="modal-stat-val">${movies.length}</div>
        <div class="modal-stat-lbl">Linked Cinema</div>
      </div>
      <div class="modal-stat-pill">
        <div class="modal-stat-val">${songs.length}</div>
        <div class="modal-stat-lbl">Linked Soundtracks</div>
      </div>
      <div class="modal-stat-pill">
        <div class="modal-stat-val">${collabs.length}</div>
        <div class="modal-stat-lbl">Creative Partners</div>
      </div>
    `;

    // Collaborations
    if (collabs.length) {
      modalCollabsList.innerHTML = collabs.map(c => `
        <div class="modal-chip" data-type="${c.role.toLowerCase()}" data-name="${escapeHtml(c.name)}">
          ${escapeHtml(c.name)} &bull; ${c.count} projects
        </div>
      `).join("");

      modalCollabsList.querySelectorAll(".modal-chip").forEach(chip => {
        chip.addEventListener("click", () => {
          openChainModal(chip.dataset.type, chip.dataset.name);
        });
      });
    } else {
      modalCollabsList.innerHTML = `<span style="color: #64748b; font-size: 12px;">No frequent partners mapped.</span>`;
    }

    // Movies
    if (movies.length) {
      modalMoviesList.innerHTML = movies.map(m => `
        <div class="modal-mini-card" data-title="${escapeHtml(m.title)}">
          <div class="mini-card-title">${escapeHtml(m.title)}</div>
          <div class="mini-card-sub">${m.release_year || ''} &bull; ${escapeHtml(m.relation || '')}</div>
        </div>
      `).join("");

      modalMoviesList.querySelectorAll(".modal-mini-card").forEach(c => {
        c.addEventListener("click", () => {
          chainModalOverlay.classList.add("hidden");
          searchInput.value = c.dataset.title;
          executeSearch(c.dataset.title);
        });
      });
    } else {
      modalMoviesList.innerHTML = `<span style="color: #64748b; font-size: 12px;">No direct movie connections.</span>`;
    }

    // Songs
    if (songs.length) {
      modalSongsList.innerHTML = songs.map(s => `
        <div class="modal-mini-card" data-title="${escapeHtml(s.title)}">
          <div class="mini-card-title">${escapeHtml(s.title)}</div>
          <div class="mini-card-sub">${escapeHtml(s.relation || s.singer || s.composer || '')}</div>
        </div>
      `).join("");

      modalSongsList.querySelectorAll(".modal-mini-card").forEach(c => {
        c.addEventListener("click", () => {
          chainModalOverlay.classList.add("hidden");
          showAudioToast(c.dataset.title, "Soundtrack");
        });
      });
    } else {
      modalSongsList.innerHTML = `<span style="color: #64748b; font-size: 12px;">No direct song connections.</span>`;
    }
  }

  // Quick entity chips inside graph modal
  document.querySelectorAll(".entity-pick-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      openChainModal(btn.dataset.type, btn.dataset.name);
    });
  });

  modalCloseBtn.addEventListener("click", () => {
    chainModalOverlay.classList.add("hidden");
    navigateTo("home");
  });

  chainModalOverlay.addEventListener("click", (e) => {
    if (e.target === chainModalOverlay) {
      chainModalOverlay.classList.add("hidden");
    }
  });

  // -------------------------------------------------------------
  // 8. AUDIO NOTIFICATION TOAST
  // -------------------------------------------------------------
  function showAudioToast(title, artist) {
    toastTitle.textContent = title;
    toastArtist.textContent = artist;
    toastPlayLink.href = `https://www.youtube.com/results?search_query=${encodeURIComponent(title + " official audio")}`;
    audioToast.classList.remove("hidden");

    setTimeout(() => {
      audioToast.classList.add("hidden");
    }, 4500);
  }

  toastClose.addEventListener("click", () => {
    audioToast.classList.add("hidden");
  });

  // -------------------------------------------------------------
  // 9. WATCH AVAILABILITY & TECHNICAL INFORMATION MODAL
  // -------------------------------------------------------------
  watchTabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      watchTabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const tabName = btn.dataset.tab;
      document.querySelectorAll(".watch-tab-content").forEach(content => {
        content.classList.remove("active");
      });
      const targetContent = document.getElementById(`tab-content-${tabName}`);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  async function openWatchModal(title, type = "movie") {
    if (!title) return;
    currentWatchTitle = title;
    currentWatchType = type;
    currentWatchData = null;

    // Reset tabs to default (first tab: streaming)
    watchTabBtns.forEach((b, i) => b.classList.toggle("active", i === 0));
    document.querySelectorAll(".watch-tab-content").forEach((c, i) => c.classList.toggle("active", i === 0));

    // Initial placeholder state
    watchModalTitle.textContent = title;
    watchModalSubtitle.textContent = `Connecting to verified streaming networks & audiovisual technical data...`;
    watchCertBadge.textContent = type === "movie" ? "U/A 16+" : "Lossless";
    watchResBadge.textContent = type === "movie" ? "4K UHD" : "24-bit Hi-Res";
    watchAudioBadge.textContent = type === "movie" ? "Dolby Atmos" : "Spatial Audio";
    renderWatchModalRating(title, type);

    if (streamingAvailabilityStatus) {
      streamingAvailabilityStatus.textContent = "Scanning certified streaming services...";
    }
    platformsGrid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 24px; color: #94a3b8; font-size: 13px;">
        <span class="stream-pulse-dot" style="display: inline-block; margin-right: 8px;"></span>
        Querying availability for <strong style="color:#22d3ee;">${escapeHtml(title)}</strong>...
      </div>
    `;
    techSpecsGrid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 24px; color: #94a3b8; font-size: 13px;">
        Retrieving technical mastering specifications...
      </div>
    `;
    soundtrackList.innerHTML = `
      <div style="text-align: center; padding: 18px; color: #94a3b8; font-size: 13px;">
        Loading soundtrack information...
      </div>
    `;

    updateWatchSaveButtonState();
    watchModalOverlay.classList.remove("hidden");

    try {
      const res = await fetch(`/api/watch-info?title=${encodeURIComponent(title)}&type=${encodeURIComponent(type)}`);
      const data = await res.json();
      currentWatchData = data;
      renderWatchModal(data);
    } catch (err) {
      console.error("Watch info error:", err);
      watchModalSubtitle.textContent = "Technical overview unavailable. Direct search active.";
      renderWatchFallback(title, type);
    }
  }

  function renderWatchModal(data) {
    const isMovie = (data.media_type || currentWatchType) === "movie";
    const specs = data.tech_specs || {};

    watchModalTitle.textContent = data.title || currentWatchTitle;

    // Subtitle meta
    if (isMovie) {
      const year = data.release_year || "Recent Release";
      const runtime = specs.runtime || "Feature Film";
      const studio = specs.studio || "Indian Cinema";
      watchModalSubtitle.textContent = `${year} • ${runtime} • ${studio}`;
    } else {
      const label = specs.label || "Record Label";
      const format = specs.format || "Hi-Res Audio";
      watchModalSubtitle.textContent = `${label} • Master Quality • ${format}`;
    }

    // Badges
    if (specs.certification) watchCertBadge.textContent = specs.certification;
    else watchCertBadge.textContent = isMovie ? "U/A 16+" : "Lossless";

    if (specs.resolution) watchResBadge.textContent = specs.resolution;
    else if (specs.format) watchResBadge.textContent = specs.format;

    if (specs.audio) watchAudioBadge.textContent = specs.audio.split(" ")[0] + " " + (specs.audio.split(" ")[1] || "Atmos");
    else if (specs.spatial_audio) watchAudioBadge.textContent = "Spatial Audio";

    // Update Rating Meter Indicator & Selection
    renderWatchModalRating(data.title || currentWatchTitle, data.media_type || currentWatchType);

    // Update Quick Hero Trailer Banner in Modal
    const heroTrailerDirectBtn = document.getElementById("hero-trailer-direct-btn");
    const heroTrailerBtnText = document.getElementById("hero-trailer-btn-text");
    const heroBannerTitle = document.getElementById("hero-banner-title");
    const heroBannerDesc = document.getElementById("hero-banner-desc");
    
    const trailerUrl = isMovie 
      ? (data.trailer_url || `https://www.youtube.com/results?search_query=${encodeURIComponent(currentWatchTitle + " official trailer")}`)
      : (data.video_song_url || data.trailer_url || `https://www.youtube.com/results?search_query=${encodeURIComponent(currentWatchTitle + " official video song")}`);
    
    if (heroTrailerDirectBtn) {
      heroTrailerDirectBtn.href = trailerUrl;
    }
    if (heroTrailerBtnText) {
      heroTrailerBtnText.textContent = isMovie ? "🎬 Watch Official Trailer" : "▶ Watch Official Video Song";
    }
    if (heroBannerTitle) {
      heroBannerTitle.textContent = isMovie ? "Watch on JioHotstar, Netflix & Leading Platforms" : "Stream on YouTube, Spotify & Hi-Res Platforms";
    }
    if (heroBannerDesc) {
      heroBannerDesc.textContent = isMovie 
        ? "Explore certified streaming partners and watch official trailer in 4K."
        : "Experience official 4K video songs and master-quality streaming audio.";
    }

    // Tab 3 label
    const tabTrailerLabel = document.getElementById("tab-trailer-label");
    if (tabTrailerLabel) {
      tabTrailerLabel.textContent = isMovie ? "🎬 Watch Trailer" : "▶ Watch Video Song";
    }

    // Platforms
    if (streamingAvailabilityStatus) {
      streamingAvailabilityStatus.textContent = data.status || "Verified Official Streaming Providers";
    }

    const platforms = data.platforms || [];
    if (platforms.length) {
      platformsGrid.innerHTML = platforms.map(p => `
        <div class="platform-card">
          <div class="platform-left">
            <div class="platform-icon-box" style="background: ${p.color || '#06b6d4'};">${p.icon || '▶'}</div>
            <div class="platform-info">
              <div class="platform-name">${escapeHtml(p.name)}</div>
              <div class="platform-tier">${escapeHtml(p.tier)}</div>
              <div class="platform-quality">${escapeHtml(p.quality)}</div>
            </div>
          </div>
          <a href="${p.url}" target="_blank" rel="noopener noreferrer" class="platform-watch-link">
            <span>${isMovie ? '▶ Watch Now' : '▶ Listen Now'}</span>
          </a>
        </div>
      `).join("");
    } else {
      platformsGrid.innerHTML = `
        <div style="grid-column: 1/-1; padding: 20px; text-align: center; color: #94a3b8;">
          Available to stream or rent via digital storefronts.
        </div>
      `;
    }

    // External aggregators
    if (linkJustwatch) linkJustwatch.href = data.justwatch_url || `https://www.justwatch.com/in/search?q=${encodeURIComponent(currentWatchTitle)}`;
    if (linkGoogleplay) linkGoogleplay.href = data.google_play_url || `https://play.google.com/store/search?q=${encodeURIComponent(currentWatchTitle)}&c=movies`;

    // Tech Specs
    const specEntries = Object.entries(specs);
    if (specEntries.length) {
      techSpecsGrid.innerHTML = specEntries.map(([key, val]) => `
        <div class="tech-spec-item">
          <span class="tech-spec-label">${formatSpecLabel(key)}</span>
          <span class="tech-spec-value">${escapeHtml(String(val))}</span>
        </div>
      `).join("");
    } else {
      techSpecsGrid.innerHTML = `<span style="color: #64748b; font-size: 12px;">Standard High Definition audiovisual masters.</span>`;
    }

    // Trailer Tab
    if (launchTrailerBtn) {
      launchTrailerBtn.href = trailerUrl;
      const span = launchTrailerBtn.querySelector("span");
      if (span) {
        span.textContent = isMovie ? "▶ Play Official Trailer on YouTube" : "▶ Play Official Music Video on YouTube";
      }
    }
    const trailerCardTitle = document.getElementById("trailer-card-title");
    const trailerCardDesc = document.getElementById("trailer-card-desc");
    if (trailerCardTitle) {
      trailerCardTitle.textContent = isMovie ? "Watch Official Cinema Trailer" : "Watch Official Music Video";
    }
    if (trailerCardDesc) {
      trailerCardDesc.textContent = isMovie
        ? "Experience the official high-definition preview, teaser, and clips directly on YouTube."
        : "Stream the official high-definition music video with 4K audiovisual presentation on YouTube.";
    }

    // Soundtracks
    const soundtracks = data.soundtracks || [];
    if (soundtracks.length) {
      soundtrackList.innerHTML = soundtracks.map(s => `
        <div class="soundtrack-item">
          <div>
            <div class="soundtrack-item-title">${escapeHtml(s.title)}</div>
            <div class="soundtrack-item-sub">${escapeHtml(s.artist || 'Original Motion Picture Soundtrack')}</div>
          </div>
          <button class="soundtrack-item-play" data-song="${escapeHtml(s.title)}">▶ Listen</button>
        </div>
      `).join("");

      soundtrackList.querySelectorAll(".soundtrack-item-play").forEach(btn => {
        btn.addEventListener("click", () => {
          showAudioToast(btn.dataset.song, currentWatchTitle);
        });
      });
    } else {
      soundtrackList.innerHTML = `
        <div class="soundtrack-item">
          <div>
            <div class="soundtrack-item-title">Original Soundtracks & Score</div>
            <div class="soundtrack-item-sub">Composed for ${escapeHtml(currentWatchTitle)}</div>
          </div>
          <button class="soundtrack-item-play" id="btn-soundtrack-search">🔍 Explore OST</button>
        </div>
      `;
      const searchOstBtn = document.getElementById("btn-soundtrack-search");
      if (searchOstBtn) {
        searchOstBtn.addEventListener("click", () => {
          watchModalOverlay.classList.add("hidden");
          searchInput.value = currentWatchTitle;
          executeSearch(currentWatchTitle);
        });
      }
    }

    // Attach fluid tilt to newly rendered platform cards
    applyFluidTiltToAll();
  }

  function renderWatchFallback(title, type) {
    const isMovie = type === "movie";
    const ytQuery = encodeURIComponent(title + (isMovie ? " full movie watch online" : " official song"));
    platformsGrid.innerHTML = `
      <div class="platform-card">
        <div class="platform-left">
          <div class="platform-icon-box" style="background: #E50914;">N</div>
          <div class="platform-info">
            <div class="platform-name">Digital Streaming Hub</div>
            <div class="platform-tier">Search Worldwide</div>
            <div class="platform-quality">High Definition</div>
          </div>
        </div>
        <a href="https://www.google.com/search?q=${ytQuery}" target="_blank" rel="noopener noreferrer" class="platform-watch-link">
          <span>Search Streaming</span>
        </a>
      </div>
    `;
    if (launchTrailerBtn) {
      launchTrailerBtn.href = `https://www.youtube.com/results?search_query=${encodeURIComponent(title + " official trailer")}`;
    }
  }

  function formatSpecLabel(key) {
    return key
      .replace(/_/g, " ")
      .replace(/\b\w/g, l => l.toUpperCase());
  }

  function updateWatchSaveButtonState() {
    if (!watchSaveText) return;
    const isSaved = isItemSaved(currentWatchTitle);
    watchSaveText.textContent = isSaved ? "✓ Saved in Collection" : "📑 Save to Collection";
  }

  // Watch Modal Footer Action Bar
  if (watchActRecommend) {
    watchActRecommend.addEventListener("click", () => {
      watchModalOverlay.classList.add("hidden");
      selectAndRecommend(currentWatchTitle, currentWatchType);
    });
  }

  if (watchActGraph) {
    watchActGraph.addEventListener("click", () => {
      watchModalOverlay.classList.add("hidden");
      openChainModal(currentWatchType, currentWatchTitle);
    });
  }

  if (watchActSave) {
    watchActSave.addEventListener("click", () => {
      toggleSaveItem(currentWatchTitle, currentWatchType, "Collection Item", null);
      updateWatchSaveButtonState();
    });
  }

  if (watchActShare) {
    watchActShare.addEventListener("click", () => {
      const shareText = `🎬 Watch "${currentWatchTitle}" on verified streaming platforms! Check Cinevibe AI: http://127.0.0.1:5000`;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(shareText).then(() => {
          showAudioToast("Watch info copied to clipboard!", currentWatchTitle);
        }).catch(() => {
          showAudioToast(currentWatchTitle, "Streaming Link Ready");
        });
      } else {
        showAudioToast(currentWatchTitle, "Streaming Link Ready");
      }
    });
  }

  watchCloseBtn.addEventListener("click", () => {
    watchModalOverlay.classList.add("hidden");
  });

  watchModalOverlay.addEventListener("click", (e) => {
    if (e.target === watchModalOverlay) {
      watchModalOverlay.classList.add("hidden");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!watchModalOverlay.classList.contains("hidden")) {
        watchModalOverlay.classList.add("hidden");
      }
      if (!chainModalOverlay.classList.contains("hidden")) {
        chainModalOverlay.classList.add("hidden");
      }
      if (!listDrawerOverlay.classList.contains("hidden")) {
        listDrawerOverlay.classList.add("hidden");
      }
      if (profileDrawerOverlay && !profileDrawerOverlay.classList.contains("hidden")) {
        profileDrawerOverlay.classList.add("hidden");
      }
      if (authModalOverlay && !authModalOverlay.classList.contains("hidden")) {
        authModalOverlay.classList.add("hidden");
      }
    }
  });

  // -------------------------------------------------------------
  // 10. USER PROFILE & AUTHENTICATION SYSTEM
  // -------------------------------------------------------------
  const navProfile = document.getElementById("nav-profile");
  const sidebarProfileTab = document.getElementById("sidebar-profile-tab");
  const sidebarProfileSettingsBtn = document.getElementById("sidebar-profile-settings-btn");
  const profileDrawerOverlay = document.getElementById("profile-drawer-overlay");
  const profileDrawerClose = document.getElementById("profile-drawer-close");
  const authModalOverlay = document.getElementById("auth-modal-overlay");
  const authCloseBtn = document.getElementById("auth-close-btn");
  const authTabLogin = document.getElementById("auth-tab-login");
  const authTabSignup = document.getElementById("auth-tab-signup");
  const authForm = document.getElementById("auth-form");
  const authNameField = document.getElementById("auth-name-field");
  const authAvatarField = document.getElementById("auth-avatar-field");
  const authInputName = document.getElementById("auth-input-name");
  const authInputEmail = document.getElementById("auth-input-email");
  const authInputPassword = document.getElementById("auth-input-password");
  const authTogglePwd = document.getElementById("auth-toggle-pwd");
  const authSubmitBtn = document.getElementById("auth-submit-btn");
  const authDemoBtn = document.getElementById("auth-demo-btn");
  const authErrorMsg = document.getElementById("auth-error-msg");
  const profileOpenLoginBtn = document.getElementById("profile-open-login-btn");
  const profileLogoutBtn = document.getElementById("profile-logout-btn");
  const profileClearDataBtn = document.getElementById("profile-clear-data-btn");
  const profileAvatarEditBadge = document.getElementById("profile-avatar-edit-badge");
  const profileEditPanel = document.getElementById("profile-edit-panel");
  const profileEditNameInput = document.getElementById("profile-edit-name-input");
  const profileSaveNameBtn = document.getElementById("profile-save-name-btn");
  const profileCancelEditBtn = document.getElementById("profile-cancel-edit-btn");
  const statCardMyList = document.getElementById("stat-card-mylist");
  const statCardFavorites = document.getElementById("stat-card-favorites");

  let authMode = "login";
  let selectedAvatar = "🎬";
  let currentUser = null;

  function loadUserSession() {
    try {
      const stored = localStorage.getItem("cinevibe_user");
      if (stored) {
        currentUser = JSON.parse(stored);
      }
    } catch (e) {
      currentUser = null;
    }

    if (!currentUser) {
      currentUser = {
        name: "Shiva",
        email: "shiva@cinevibe.ai",
        avatar: "🎬",
        role: "VIP Cinephile",
        joined: "Active 2026",
        loggedIn: true
      };
      saveUserSession();
    }
    selectedAvatar = currentUser.avatar || "🎬";
    updateProfileUI();
  }

  function saveUserSession() {
    try {
      localStorage.setItem("cinevibe_user", JSON.stringify(currentUser));
    } catch (e) {
      console.error("Failed to save user session", e);
    }
  }

  function updateProfileUI() {
    if (!currentUser) return;

    // Sidebar left-bottom corner
    const avatarText = document.getElementById("sidebar-avatar-text");
    const profileName = document.getElementById("sidebar-profile-name");
    const profileRole = document.getElementById("sidebar-profile-role");
    if (avatarText) avatarText.textContent = currentUser.avatar || "🎬";
    if (profileName) profileName.textContent = currentUser.name || "Guest";
    if (profileRole) profileRole.textContent = currentUser.loggedIn ? (currentUser.role || "VIP Cinephile") : "Guest Explorer";

    // Drawer elements
    const drawerAvatar = document.getElementById("profile-modal-avatar");
    const drawerName = document.getElementById("profile-modal-name");
    const drawerEmail = document.getElementById("profile-modal-email");
    const drawerTier = document.getElementById("profile-modal-tier");
    const drawerJoined = document.getElementById("profile-modal-joined");

    if (drawerAvatar) drawerAvatar.textContent = currentUser.avatar || "🎬";
    if (drawerName) drawerName.textContent = currentUser.name || "Guest";
    if (drawerEmail) drawerEmail.textContent = currentUser.email || "guest@cinevibe.ai";
    if (drawerTier) drawerTier.textContent = currentUser.loggedIn ? (currentUser.role || "🌟 VIP Cinephile") : "Guest Explorer";
    if (drawerJoined) drawerJoined.textContent = currentUser.joined || "Active 2026";

    // Live engagement stats
    const statMyList = document.getElementById("profile-stat-mylist");
    const statFavorites = document.getElementById("profile-stat-favorites");
    const statRatings = document.getElementById("profile-stat-ratings");
    if (statMyList) statMyList.textContent = myList.length;
    if (statFavorites) statFavorites.textContent = myFavorites.length;
    if (statRatings) statRatings.textContent = Object.keys(getUserRatings()).length;

    // Pre-populate inline name input
    if (profileEditNameInput) {
      profileEditNameInput.value = currentUser.name || "Shiva";
    }
  }

  function openProfileDrawer() {
    updateProfileUI();
    if (profileDrawerOverlay) profileDrawerOverlay.classList.remove("hidden");
  }

  function closeProfileDrawer() {
    if (profileDrawerOverlay) profileDrawerOverlay.classList.add("hidden");
    if (profileEditPanel) profileEditPanel.classList.add("hidden");
  }

  function setAuthMode(mode) {
    authMode = mode;
    if (authErrorMsg) authErrorMsg.classList.add("hidden");

    if (mode === "signup") {
      if (authTabSignup) authTabSignup.classList.add("active");
      if (authTabLogin) authTabLogin.classList.remove("active");
      if (authNameField) authNameField.style.display = "flex";
      if (authAvatarField) authAvatarField.style.display = "flex";
      if (authSubmitBtn) authSubmitBtn.querySelector("span").textContent = "Create Cinevibe Account";
      const title = document.getElementById("auth-title");
      if (title) title.textContent = "Join Cinevibe AI";
    } else {
      if (authTabLogin) authTabLogin.classList.add("active");
      if (authTabSignup) authTabSignup.classList.remove("active");
      if (authNameField) authNameField.style.display = "none";
      if (authAvatarField) authAvatarField.style.display = "none";
      if (authSubmitBtn) authSubmitBtn.querySelector("span").textContent = "Sign In to Cinevibe";
      const title = document.getElementById("auth-title");
      if (title) title.textContent = "Welcome to Cinevibe AI";
    }
  }

  function openAuthModal(isSignup = false) {
    closeProfileDrawer();
    setAuthMode(isSignup ? "signup" : "login");
    if (authModalOverlay) authModalOverlay.classList.remove("hidden");
  }

  function closeAuthModal() {
    if (authModalOverlay) authModalOverlay.classList.add("hidden");
    if (authErrorMsg) authErrorMsg.classList.add("hidden");
  }

  // Navigation Profile click
  if (navProfile) {
    navProfile.addEventListener("click", (e) => {
      e.preventDefault();
      navigateTo("profile");
    });
  }

  // Left-bottom corner profile tab click
  if (sidebarProfileTab) {
    sidebarProfileTab.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      navigateTo("profile");
    });
  }

  if (sidebarProfileSettingsBtn) {
    sidebarProfileSettingsBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      navigateTo("profile");
      if (profileEditPanel) profileEditPanel.classList.remove("hidden");
    });
  }

  // Profile Drawer close
  if (profileDrawerClose) {
    profileDrawerClose.addEventListener("click", () => {
      closeProfileDrawer();
      navigateTo("home");
    });
  }
  if (profileDrawerOverlay) {
    profileDrawerOverlay.addEventListener("click", (e) => {
      if (e.target === profileDrawerOverlay) closeProfileDrawer();
    });
  }

  // Stat cards jump to My List & Favorites drawers
  if (statCardMyList) {
    statCardMyList.addEventListener("click", () => {
      closeProfileDrawer();
      openDrawer("list");
    });
  }
  if (statCardFavorites) {
    statCardFavorites.addEventListener("click", () => {
      closeProfileDrawer();
      openDrawer("favorites");
    });
  }

  // Inline edit toggle
  if (profileAvatarEditBadge) {
    profileAvatarEditBadge.addEventListener("click", () => {
      if (profileEditPanel) {
        profileEditPanel.classList.toggle("hidden");
      }
    });
  }

  if (profileSaveNameBtn) {
    profileSaveNameBtn.addEventListener("click", () => {
      const newName = (profileEditNameInput.value || "").trim();
      if (newName) {
        currentUser.name = newName;
        currentUser.avatar = selectedAvatar;
        saveUserSession();
        updateProfileUI();
        if (profileEditPanel) profileEditPanel.classList.add("hidden");
        showAudioToast(`Profile updated to ${newName}!`, "Account Synced");
      }
    });
  }

  if (profileCancelEditBtn) {
    profileCancelEditBtn.addEventListener("click", () => {
      if (profileEditPanel) profileEditPanel.classList.add("hidden");
    });
  }

  // Profile avatar selector buttons
  document.querySelectorAll("#profile-avatar-picker-panel .avatar-opt-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#profile-avatar-picker-panel .avatar-opt-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      selectedAvatar = btn.dataset.avatar || "🎬";
      currentUser.avatar = selectedAvatar;
      saveUserSession();
      updateProfileUI();
      showAudioToast(`Persona avatar set to ${selectedAvatar}`, "Profile Updated");
    });
  });

  // Open login from profile drawer
  if (profileOpenLoginBtn) {
    profileOpenLoginBtn.addEventListener("click", () => {
      openAuthModal(false);
    });
  }

  // Log Out
  if (profileLogoutBtn) {
    profileLogoutBtn.addEventListener("click", () => {
      currentUser.loggedIn = false;
      currentUser.name = "Guest";
      currentUser.role = "Guest Explorer";
      saveUserSession();
      updateProfileUI();
      closeProfileDrawer();
      showAudioToast("Signed out. Exploring as Guest.", "Logged Out");
    });
  }

  // Clear Saved Data
  if (profileClearDataBtn) {
    profileClearDataBtn.addEventListener("click", () => {
      if (confirm("Clear all items from your Saved List and Favorites?")) {
        myList = [];
        myFavorites = [];
        localStorage.setItem("cinevibe_my_list", "[]");
        localStorage.setItem("velora_my_list", "[]");
        localStorage.setItem("cinevibe_favorites", "[]");
        localStorage.setItem("velora_favorites", "[]");
        updateBadges();
        updateProfileUI();
        showAudioToast("Your personal collection has been reset.", "Collection Cleared");
      }
    });
  }

  // Auth Modal events
  if (authCloseBtn) authCloseBtn.addEventListener("click", closeAuthModal);
  if (authModalOverlay) {
    authModalOverlay.addEventListener("click", (e) => {
      if (e.target === authModalOverlay) closeAuthModal();
    });
  }

  if (authTabLogin) authTabLogin.addEventListener("click", () => setAuthMode("login"));
  if (authTabSignup) authTabSignup.addEventListener("click", () => setAuthMode("signup"));

  // Auth avatar options in signup
  document.querySelectorAll("#auth-avatar-options .avatar-opt-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#auth-avatar-options .avatar-opt-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      selectedAvatar = btn.dataset.avatar || "🎬";
    });
  });

  // Password visibility toggle
  if (authTogglePwd) {
    authTogglePwd.addEventListener("click", () => {
      if (authInputPassword.type === "password") {
        authInputPassword.type = "text";
        authTogglePwd.textContent = "🙈";
      } else {
        authInputPassword.type = "password";
        authTogglePwd.textContent = "👁️";
      }
    });
  }

  // Quick Demo Login (Shiva)
  if (authDemoBtn) {
    authDemoBtn.addEventListener("click", () => {
      currentUser = {
        name: "Shiva",
        email: "shiva@cinevibe.ai",
        avatar: "🎬",
        role: "VIP Cinephile",
        joined: "Active 2026",
        loggedIn: true
      };
      saveUserSession();
      updateProfileUI();
      closeAuthModal();
      showAudioToast("Welcome back, Shiva! VIP status active.", "Cinevibe AI");
    });
  }

  // Auth Form Submit
  if (authForm) {
    authForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const email = (authInputEmail.value || "").trim();
      const pwd = (authInputPassword.value || "").trim();

      if (!email || !pwd) {
        if (authErrorMsg) {
          authErrorMsg.textContent = "Please fill in all required fields.";
          authErrorMsg.classList.remove("hidden");
        }
        return;
      }

      let name = "Shiva";
      if (authMode === "signup") {
        name = (authInputName.value || "").trim() || email.split("@")[0] || "Cinephile";
      } else {
        name = email.split("@")[0].replace(/[._-]/g, " ").replace(/\b\w/g, l => l.toUpperCase()) || "Shiva";
      }

      currentUser = {
        name: name,
        email: email,
        avatar: selectedAvatar || "🎬",
        role: "VIP Cinephile",
        joined: "Active 2026",
        loggedIn: true
      };
      saveUserSession();
      updateProfileUI();
      closeAuthModal();
      showAudioToast(`Welcome to Cinevibe, ${name}!`, "Account Ready");
    });
  }

  // Load session on startup
  loadUserSession();

  // Handle initial route on startup if hash is present
  const initialHash = window.location.hash;
  if (initialHash && ["#discover", "#graph", "#mylist", "#list", "#favorites", "#profile"].includes(initialHash.toLowerCase())) {
    navigateTo(initialHash, false);
  }

  // -------------------------------------------------------------
  // 10. ULTRA-FLUID 3D MAGNETIC CARD TILT PHYSICS & DYNAMICS
  // -------------------------------------------------------------
  function attachFluid3DTilt(card) {
    if (!card || card._tiltAttached) return;
    card._tiltAttached = true;

    card.addEventListener("pointermove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      const rotateX = -y * 11;
      const rotateY = x * 11;

      card.style.transform = `perspective(850px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateY(-5px) scale3d(1.025, 1.025, 1.025)`;
      card.style.transition = 'transform 0.08s ease-out, box-shadow 0.15s ease';
    });

    card.addEventListener("pointerleave", () => {
      card.style.transform = 'perspective(850px) rotateX(0deg) rotateY(0deg) translateY(0px) scale3d(1, 1, 1)';
      card.style.transition = 'transform 0.45s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.4s ease';
    });
  }

  function applyFluidTiltToAll() {
    document.querySelectorAll(".poster-card, .dual-card, .entity-item, .quick-tile, .platform-card").forEach(attachFluid3DTilt);
  }

  // Initial tilt application across all cards
  applyFluidTiltToAll();
});
