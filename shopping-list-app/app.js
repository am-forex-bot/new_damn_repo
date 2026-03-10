// ===== The Shopping List - App Logic =====

(function () {
  'use strict';

  // ---- State ----
  const DEFAULT_STORES = ['Aldi', "Sainsbury's"];
  let state = {
    listId: null,
    stores: [...DEFAULT_STORES],
    splitAisles: {}, // { "Sainsbury's": true }
    activeStore: DEFAULT_STORES[0],
    items: [],       // { id, name, store, completed, addedAt, aisleInfo }
    aisleMemory: {}, // { "store::itemNormalized": { aisle, half, side, position } }
  };

  let db = null; // Firebase database reference
  let unsubscribe = null; // Firestore listener

  // ---- DOM Refs ----
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const els = {
    tabs: $('#tabs'),
    list: $('#shopping-list'),
    emptyState: $('#empty-state'),
    textInput: $('#text-input'),
    btnAdd: $('#btn-add'),
    btnMic: $('#btn-mic'),
    btnAddStore: $('#btn-add-store'),
    btnSettings: $('#btn-settings'),
    voiceOverlay: $('#voice-overlay'),
    voiceStatus: $('#voice-status'),
    voiceTranscript: $('#voice-transcript'),
    btnVoiceCancel: $('#btn-voice-cancel'),
    aisleModal: $('#aisle-modal'),
    aisleItemName: $('#aisle-item-name'),
    aisleNumber: $('#aisle-number'),
    aisleHalfSection: $('#aisle-half-section'),
    btnAisleSkip: $('#btn-aisle-skip'),
    btnAisleSave: $('#btn-aisle-save'),
    settingsModal: $('#settings-modal'),
    storeListSettings: $('#store-list-settings'),
    splitAisleToggles: $('#split-aisle-toggles'),
    syncStatus: $('#sync-status'),
    listIdInput: $('#list-id-input'),
    btnGenerateId: $('#btn-generate-id'),
    btnApplyId: $('#btn-apply-id'),
    btnCopyId: $('#btn-copy-id'),
    btnWhatsappShare: $('#btn-whatsapp-share'),
    btnClearCompleted: $('#btn-clear-completed'),
    btnSettingsClose: $('#btn-settings-close'),
    addStoreModal: $('#add-store-modal'),
    newStoreName: $('#new-store-name'),
    btnStoreCancel: $('#btn-store-cancel'),
    btnStoreSave: $('#btn-store-save'),
  };

  // ---- Local Storage ----
  function loadLocalState() {
    try {
      const saved = localStorage.getItem('shoppingListState');
      if (saved) {
        const parsed = JSON.parse(saved);
        state = { ...state, ...parsed };
        // Ensure defaults exist
        if (!state.stores.length) state.stores = [...DEFAULT_STORES];
        if (!state.activeStore) state.activeStore = state.stores[0];
        if (!state.splitAisles) state.splitAisles = {};
        if (!state.aisleMemory) state.aisleMemory = {};
      }
    } catch (e) { /* ignore */ }
  }

  function saveLocalState() {
    try {
      localStorage.setItem('shoppingListState', JSON.stringify(state));
    } catch (e) { /* ignore */ }
  }

  // ---- Firebase / Sync ----
  function initFirebase() {
    if (typeof firebase === 'undefined' || !window.FIREBASE_CONFIG) {
      updateSyncStatus(false);
      return;
    }

    try {
      if (!firebase.apps.length) {
        firebase.initializeApp(window.FIREBASE_CONFIG);
      }
      db = firebase.firestore();

      if (state.listId) {
        startSync();
      } else {
        updateSyncStatus(false);
      }
    } catch (e) {
      console.error('Firebase init error:', e);
      updateSyncStatus(false);
    }
  }

  function startSync() {
    if (!db || !state.listId) return;
    stopSync();

    const docRef = db.collection('shoppingLists').doc(state.listId);

    // Listen for real-time changes
    unsubscribe = docRef.onSnapshot((doc) => {
      if (doc.exists) {
        const data = doc.data();
        state.items = data.items || [];
        state.stores = data.stores || state.stores;
        state.splitAisles = data.splitAisles || {};
        state.aisleMemory = data.aisleMemory || {};
        // Keep activeStore local
        if (!state.stores.includes(state.activeStore)) {
          state.activeStore = state.stores[0];
        }
        saveLocalState();
        renderTabs();
        renderList();
      }
      updateSyncStatus(true);
    }, (error) => {
      console.error('Sync error:', error);
      updateSyncStatus(false);
    });
  }

  function stopSync() {
    if (unsubscribe) {
      unsubscribe();
      unsubscribe = null;
    }
  }

  function syncToCloud() {
    if (!db || !state.listId) {
      saveLocalState();
      return;
    }

    const docRef = db.collection('shoppingLists').doc(state.listId);
    docRef.set({
      items: state.items,
      stores: state.stores,
      splitAisles: state.splitAisles,
      aisleMemory: state.aisleMemory,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
    }, { merge: true }).catch((e) => {
      console.error('Sync write error:', e);
      toast('Sync failed - changes saved locally');
    });

    saveLocalState();
  }

  function updateSyncStatus(connected) {
    els.syncStatus.textContent = connected ? 'Connected & syncing' : 'Offline (local only)';
    els.syncStatus.className = 'sync-indicator ' + (connected ? 'connected' : 'disconnected');
  }

  function generateListId() {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    let id = '';
    for (let i = 0; i < 8; i++) id += chars[Math.floor(Math.random() * chars.length)];
    return id;
  }

  // ---- Rendering ----
  function renderTabs() {
    els.tabs.innerHTML = '';

    // "All" tab
    const allCount = state.items.filter(i => !i.completed).length;
    const allBtn = createTab('All', allCount, state.activeStore === 'All');
    allBtn.addEventListener('click', () => {
      state.activeStore = 'All';
      saveLocalState();
      renderTabs();
      renderList();
    });
    els.tabs.appendChild(allBtn);

    state.stores.forEach(store => {
      const count = state.items.filter(i => i.store === store && !i.completed).length;
      const btn = createTab(store, count, state.activeStore === store);
      btn.addEventListener('click', () => {
        state.activeStore = store;
        saveLocalState();
        renderTabs();
        renderList();
      });
      els.tabs.appendChild(btn);
    });
  }

  function createTab(name, count, active) {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (active ? ' active' : '');
    btn.innerHTML = `${name}<span class="tab-count">${count || ''}</span>`;
    return btn;
  }

  function renderList() {
    let items = state.items;
    if (state.activeStore !== 'All') {
      items = items.filter(i => i.store === state.activeStore);
    }

    // Sort: uncompleted first, then by addedAt
    items.sort((a, b) => {
      if (a.completed !== b.completed) return a.completed ? 1 : -1;
      // Sort uncompleted by aisle info for smart ordering
      if (!a.completed && !b.completed && a.aisleInfo && b.aisleInfo) {
        const aKey = aisleSort(a.aisleInfo);
        const bKey = aisleSort(b.aisleInfo);
        if (aKey !== bKey) return aKey - bKey;
      }
      return (a.addedAt || 0) - (b.addedAt || 0);
    });

    if (items.length === 0) {
      els.emptyState.classList.remove('hidden');
      els.list.innerHTML = '';
      return;
    }

    els.emptyState.classList.add('hidden');
    els.list.innerHTML = '';

    // If showing "All", group by store
    if (state.activeStore === 'All') {
      const grouped = {};
      items.forEach(item => {
        if (!grouped[item.store]) grouped[item.store] = [];
        grouped[item.store].push(item);
      });
      Object.keys(grouped).forEach(store => {
        const header = document.createElement('li');
        header.className = 'list-item';
        header.style.background = 'transparent';
        header.style.padding = '8px 4px 4px';
        header.innerHTML = `<span style="font-weight:700;color:var(--accent);font-size:13px;text-transform:uppercase;letter-spacing:0.5px">${store}</span>`;
        els.list.appendChild(header);
        grouped[store].forEach(item => els.list.appendChild(createItemEl(item)));
      });
    } else {
      items.forEach(item => els.list.appendChild(createItemEl(item)));
    }
  }

  function aisleSort(info) {
    if (!info) return 9999;
    let n = (info.aisle || 0) * 100;
    if (info.half === 'B') n += 50;
    n += (info.position || 0) * 10;
    if (info.side === 'R') n += 5;
    return n;
  }

  function createItemEl(item) {
    const li = document.createElement('li');
    li.className = 'list-item' + (item.completed ? ' completed' : '');
    li.dataset.id = item.id;

    const aisleText = formatAisleInfo(item.aisleInfo);
    const aisleHtml = aisleText ? `<div class="item-aisle">${aisleText}</div>` : '';

    li.innerHTML = `
      <div class="item-checkbox ${item.completed ? 'checked' : ''}" data-action="toggle"></div>
      <div class="item-info">
        <div class="item-name">${escapeHtml(item.name)}</div>
        ${aisleHtml}
      </div>
      <button class="item-delete" data-action="delete" aria-label="Delete">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
    `;

    li.querySelector('[data-action="toggle"]').addEventListener('click', () => toggleItem(item.id));
    li.querySelector('[data-action="delete"]').addEventListener('click', () => deleteItem(item.id));

    return li;
  }

  function formatAisleInfo(info) {
    if (!info) return '';
    const sides = { L: 'left', R: 'right' };
    const positions = { 1: 'start', 2: 'middle', 3: 'end' };
    let text = `Aisle ${info.aisle}`;
    if (info.half) text += info.half;
    text += ` \u2022 ${sides[info.side] || '?'} side`;
    text += ` \u2022 ${positions[info.position] || '?'}`;
    return text;
  }

  // ---- Item Actions ----
  function addItem(rawText) {
    if (!rawText || !rawText.trim()) return;

    const parsed = parseInput(rawText.trim());
    if (!parsed.name) return;

    // Determine store
    let store = parsed.store || state.activeStore;
    if (store === 'All') store = state.stores[0];

    // Add store if it doesn't exist
    if (!state.stores.includes(store)) {
      state.stores.push(store);
    }

    // Duplicate detection
    const normalized = normalizeItemName(parsed.name);
    const duplicate = state.items.find(i =>
      i.store === store &&
      !i.completed &&
      normalizeItemName(i.name) === normalized
    );

    if (duplicate) {
      toast(`"${parsed.name}" is already on the ${store} list`);
      return;
    }

    // Check aisle memory
    const memoryKey = `${store}::${normalized}`;
    const rememberedAisle = state.aisleMemory[memoryKey] || null;

    const item = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      name: parsed.name,
      store: store,
      completed: false,
      addedAt: Date.now(),
      aisleInfo: rememberedAisle,
    };

    state.items.push(item);
    state.activeStore = store;
    syncToCloud();
    renderTabs();
    renderList();

    if (rememberedAisle) {
      toast(`Added "${parsed.name}" - ${formatAisleInfo(rememberedAisle)}`);
    } else {
      toast(`Added "${parsed.name}" to ${store}`);
    }
  }

  function parseInput(text) {
    // Patterns to detect: "item - store", "item store", "item, store"
    // Try "item - store" first
    let match = text.match(/^(.+?)\s*[-\u2013\u2014]\s*(.+)$/);
    if (match) {
      const store = matchStore(match[2].trim());
      if (store) return { name: cleanItemName(match[1].trim()), store };
    }

    // Try "item, store"
    match = text.match(/^(.+?)\s*,\s*(.+)$/);
    if (match) {
      const store = matchStore(match[2].trim());
      if (store) return { name: cleanItemName(match[1].trim()), store };
    }

    // Try if the last word is a store name
    const words = text.split(/\s+/);
    if (words.length > 1) {
      const lastWord = words[words.length - 1];
      const store = matchStore(lastWord);
      if (store) {
        return { name: cleanItemName(words.slice(0, -1).join(' ')), store };
      }
      // Try last two words
      if (words.length > 2) {
        const lastTwo = words.slice(-2).join(' ');
        const store2 = matchStore(lastTwo);
        if (store2) {
          return { name: cleanItemName(words.slice(0, -2).join(' ')), store: store2 };
        }
      }
    }

    return { name: cleanItemName(text), store: null };
  }

  function matchStore(text) {
    const lower = text.toLowerCase().replace(/['']/g, "'");
    // Exact match
    for (const s of state.stores) {
      if (s.toLowerCase() === lower) return s;
    }
    // Partial / fuzzy match
    for (const s of state.stores) {
      if (s.toLowerCase().startsWith(lower) || lower.startsWith(s.toLowerCase())) return s;
    }
    // Common aliases
    const aliases = {
      "sainsbury": "Sainsbury's", "sainsburys": "Sainsbury's", "sainsbury's": "Sainsbury's",
      "sainos": "Sainsbury's", "sains": "Sainsbury's",
      "aldi": "Aldi", "aldis": "Aldi",
      "lidl": "Lidl", "lidls": "Lidl",
      "tesco": "Tesco", "tescos": "Tesco",
      "asda": "Asda", "asdas": "Asda",
      "morrisons": "Morrisons", "morrison": "Morrisons", "morries": "Morrisons",
      "waitrose": "Waitrose",
      "coop": "Co-op", "co-op": "Co-op",
      "m&s": "M&S", "marks": "M&S",
    };
    const alias = aliases[lower];
    if (alias) {
      // Add to stores if not present
      if (!state.stores.includes(alias)) state.stores.push(alias);
      return alias;
    }
    // If it looks like a store name (capitalized), create it
    if (text.length > 2 && text[0] === text[0].toUpperCase()) {
      if (!state.stores.includes(text)) state.stores.push(text);
      return text;
    }
    return null;
  }

  function cleanItemName(name) {
    // Capitalize first letter, trim
    name = name.trim();
    if (name.length === 0) return name;
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  function normalizeItemName(name) {
    return name.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
      // Common variations
      .replace(/^(tinned|canned|tin of|can of)\s+/, '')
      .replace(/\s*(x\s*\d+|\d+\s*x)$/, '') // remove "x2", "2x"
      .replace(/\s*(pack|bag|box|bottle|jar|tin|can)s?$/, '')
      ;
  }

  function toggleItem(id) {
    const item = state.items.find(i => i.id === id);
    if (!item) return;

    item.completed = !item.completed;

    if (item.completed && !item.aisleInfo) {
      // Show aisle picker
      showAislePicker(item);
    }

    syncToCloud();
    renderTabs();
    renderList();
  }

  function deleteItem(id) {
    state.items = state.items.filter(i => i.id !== id);
    syncToCloud();
    renderTabs();
    renderList();
  }

  // ---- Aisle Picker ----
  let currentAisleItem = null;

  function showAislePicker(item) {
    currentAisleItem = item;
    els.aisleItemName.textContent = item.name;
    els.aisleNumber.value = 1;

    // Show/hide split aisle option
    const hasSplit = state.splitAisles[item.store];
    if (hasSplit) {
      els.aisleHalfSection.classList.remove('hidden');
    } else {
      els.aisleHalfSection.classList.add('hidden');
    }

    // Reset toggles
    els.aisleModal.querySelectorAll('.toggle-group').forEach(group => {
      const btns = group.querySelectorAll('.toggle-btn');
      btns.forEach((b, i) => {
        // Default: first for side, second (middle) for position
        if (group.classList.contains('triple')) {
          b.classList.toggle('active', i === 1);
        } else {
          b.classList.toggle('active', i === 0);
        }
      });
    });

    els.aisleModal.classList.remove('hidden');
  }

  function hideAislePicker() {
    els.aisleModal.classList.add('hidden');
    currentAisleItem = null;
  }

  function saveAisleInfo() {
    if (!currentAisleItem) return;

    const groups = els.aisleModal.querySelectorAll('.toggle-group');
    let half = null;
    let side = 'L';
    let position = '2';

    const hasSplit = state.splitAisles[currentAisleItem.store];

    groups.forEach((group, idx) => {
      const active = group.querySelector('.toggle-btn.active');
      if (!active) return;
      if (hasSplit && idx === 0) half = active.dataset.value;
      else if ((!hasSplit && idx === 0) || (hasSplit && idx === 1)) side = active.dataset.value;
      else position = active.dataset.value;
    });

    const info = {
      aisle: parseInt(els.aisleNumber.value) || 1,
      half: half,
      side: side,
      position: parseInt(position) || 2,
    };

    currentAisleItem.aisleInfo = info;

    // Save to memory for future
    const normalized = normalizeItemName(currentAisleItem.name);
    const memoryKey = `${currentAisleItem.store}::${normalized}`;
    state.aisleMemory[memoryKey] = info;

    syncToCloud();
    renderList();
    hideAislePicker();
    toast(`Saved - ${formatAisleInfo(info)}`);
  }

  // ---- Voice Input ----
  let recognition = null;

  function startVoice() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast('Voice input not supported on this browser');
      return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = 'en-GB';
    recognition.continuous = false;
    recognition.interimResults = true;

    els.voiceOverlay.classList.remove('hidden');
    els.voiceStatus.textContent = 'Listening...';
    els.voiceTranscript.textContent = '';
    els.btnMic.classList.add('listening');

    recognition.onresult = (event) => {
      let interim = '';
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          final += event.results[i][0].transcript;
        } else {
          interim += event.results[i][0].transcript;
        }
      }
      els.voiceTranscript.textContent = final || interim;
      if (final) {
        els.voiceStatus.textContent = 'Got it!';
        setTimeout(() => {
          hideVoice();
          addItem(final);
        }, 500);
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech error:', event.error);
      if (event.error === 'no-speech') {
        els.voiceStatus.textContent = 'No speech detected, try again';
      } else {
        els.voiceStatus.textContent = 'Error - try again';
      }
      setTimeout(hideVoice, 1500);
    };

    recognition.onend = () => {
      if (els.voiceOverlay.classList.contains('hidden')) return;
      // If no final result, hide after a moment
      setTimeout(() => {
        if (!els.voiceOverlay.classList.contains('hidden')) {
          hideVoice();
        }
      }, 1000);
    };

    recognition.start();
  }

  function hideVoice() {
    els.voiceOverlay.classList.add('hidden');
    els.btnMic.classList.remove('listening');
    if (recognition) {
      try { recognition.stop(); } catch (e) { /* ignore */ }
      recognition = null;
    }
  }

  // ---- Event Binding ----
  function bindEvents() {
    // Add item
    els.btnAdd.addEventListener('click', () => {
      addItem(els.textInput.value);
      els.textInput.value = '';
    });

    els.textInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        addItem(els.textInput.value);
        els.textInput.value = '';
      }
    });

    // Voice
    els.btnMic.addEventListener('click', startVoice);
    els.btnVoiceCancel.addEventListener('click', hideVoice);

    // Aisle picker
    els.aisleModal.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const group = btn.closest('.toggle-group');
        group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    els.aisleModal.querySelectorAll('.aisle-num-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const delta = parseInt(btn.dataset.delta);
        const input = els.aisleNumber;
        input.value = Math.max(1, (parseInt(input.value) || 1) + delta);
      });
    });

    els.btnAisleSkip.addEventListener('click', hideAislePicker);
    els.btnAisleSave.addEventListener('click', saveAisleInfo);
    els.aisleModal.querySelector('.modal-backdrop').addEventListener('click', hideAislePicker);

    // Settings
    els.btnSettings.addEventListener('click', showSettings);
    els.btnSettingsClose.addEventListener('click', hideSettings);
    els.settingsModal.querySelector('.modal-backdrop').addEventListener('click', hideSettings);

    els.btnGenerateId.addEventListener('click', () => {
      els.listIdInput.value = generateListId();
    });

    els.btnApplyId.addEventListener('click', () => {
      const id = els.listIdInput.value.trim().toLowerCase();
      if (!id) { toast('Enter a list ID'); return; }
      state.listId = id;
      saveLocalState();
      startSync();
      toast('List ID applied - syncing!');
    });

    els.btnCopyId.addEventListener('click', () => {
      if (!state.listId) { toast('Generate or enter a list ID first'); return; }
      navigator.clipboard.writeText(state.listId).then(() => {
        toast('List ID copied! Share it with your partner');
      }).catch(() => {
        // Fallback
        els.listIdInput.value = state.listId;
        els.listIdInput.select();
        toast('Copy the ID from the field above');
      });
    });

    els.btnWhatsappShare.addEventListener('click', () => {
      if (!state.listId) { toast('Generate or enter a list ID first'); return; }
      const appUrl = window.location.href;
      const msg = `Hey! Here's our shared shopping list.\n\n` +
        `1. Open the app: ${appUrl}\n` +
        `2. Tap the ⚙ settings icon (top right)\n` +
        `3. Paste this List ID: ${state.listId}\n` +
        `4. Tap "Apply"\n\n` +
        `We're synced! 🛒`;
      const waUrl = `https://wa.me/?text=${encodeURIComponent(msg)}`;
      window.open(waUrl, '_blank');
    });

    els.btnClearCompleted.addEventListener('click', () => {
      const count = state.items.filter(i => i.completed).length;
      if (count === 0) { toast('No completed items to clear'); return; }
      state.items = state.items.filter(i => !i.completed);
      syncToCloud();
      renderTabs();
      renderList();
      toast(`Cleared ${count} item${count > 1 ? 's' : ''}`);
    });

    // Add store modal
    els.btnAddStore.addEventListener('click', () => {
      els.addStoreModal.classList.remove('hidden');
      els.newStoreName.value = '';
      els.newStoreName.focus();
    });

    els.btnStoreCancel.addEventListener('click', () => els.addStoreModal.classList.add('hidden'));
    els.addStoreModal.querySelector('.modal-backdrop').addEventListener('click', () => els.addStoreModal.classList.add('hidden'));

    els.btnStoreSave.addEventListener('click', () => {
      const name = els.newStoreName.value.trim();
      if (!name) return;
      if (state.stores.includes(name)) {
        toast('Store already exists');
        return;
      }
      state.stores.push(name);
      state.activeStore = name;
      syncToCloud();
      renderTabs();
      renderList();
      els.addStoreModal.classList.add('hidden');
      toast(`Added ${name}`);
    });

    els.newStoreName.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') els.btnStoreSave.click();
    });
  }

  // ---- Settings ----
  function showSettings() {
    els.listIdInput.value = state.listId || '';

    // Render store list
    els.storeListSettings.innerHTML = '';
    state.stores.forEach(store => {
      const row = document.createElement('div');
      row.className = 'store-setting-row';
      row.innerHTML = `
        <span>${escapeHtml(store)}</span>
        <button class="delete-store-btn" data-store="${escapeHtml(store)}">Remove</button>
      `;
      row.querySelector('.delete-store-btn').addEventListener('click', () => {
        if (state.stores.length <= 1) { toast("Can't remove the last store"); return; }
        const itemCount = state.items.filter(i => i.store === store).length;
        if (itemCount > 0 && !confirm(`Remove "${store}" and its ${itemCount} items?`)) return;
        state.stores = state.stores.filter(s => s !== store);
        state.items = state.items.filter(i => i.store !== store);
        delete state.splitAisles[store];
        if (state.activeStore === store) state.activeStore = state.stores[0];
        syncToCloud();
        showSettings(); // re-render
        renderTabs();
        renderList();
      });
      els.storeListSettings.appendChild(row);
    });

    // Render split aisle toggles
    els.splitAisleToggles.innerHTML = '';
    state.stores.forEach(store => {
      const row = document.createElement('div');
      row.className = 'split-toggle-row';
      const checked = state.splitAisles[store] ? 'checked' : '';
      row.innerHTML = `
        <span>${escapeHtml(store)}</span>
        <label class="switch">
          <input type="checkbox" ${checked} data-store="${escapeHtml(store)}">
          <span class="slider"></span>
        </label>
      `;
      row.querySelector('input').addEventListener('change', (e) => {
        state.splitAisles[store] = e.target.checked;
        syncToCloud();
      });
      els.splitAisleToggles.appendChild(row);
    });

    els.settingsModal.classList.remove('hidden');
  }

  function hideSettings() {
    els.settingsModal.classList.add('hidden');
  }

  // ---- Utilities ----
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function toast(msg) {
    // Remove existing toasts
    document.querySelectorAll('.toast').forEach(t => t.remove());
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
  }

  function registerServiceWorker() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('sw.js').catch(() => {});
    }
  }

  // ---- Boot ----
  document.addEventListener('DOMContentLoaded', () => {
    loadLocalState();
    renderTabs();
    renderList();
    bindEvents();
    registerServiceWorker();

    // Firebase loads async - init when ready
    if (window.FIREBASE_CONFIG) {
      document.addEventListener('firebase-ready', () => {
        initFirebase();
      });
    } else {
      updateSyncStatus(false);
    }
  });
})();
