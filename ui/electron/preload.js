const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('meetmindDesktop', {
  platform: process.platform,
  appMode: process.env.MEETMIND_AIR_GAPPED === '1' ? 'air-gapped' : 'local',
});
