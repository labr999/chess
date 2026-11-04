# 象棋 AI 語音＋棋盤辨識 覆盤系統（Pikafish 即時分析）

影片解說同步的象棋覆盤網頁。每切換一個節點，Pikafish 會在瀏覽器裡即時分析：

- 紅黑優勢分數條
- 綠色箭頭標出最佳著法
- 1／3／5 條主要變化，以中文棋譜顯示（例：炮二平五 馬８進７）
- 自動判斷輪到哪方，可手動切換

## 執行方式
- **網頁版（GitHub Pages）**：直接開啟網址。第一次會下載約 50 MB 神經網路權重，之後由瀏覽器快取。建議使用 Chrome／Edge 桌機版。
- **本機版**：搭配 server.py 與原生 Pikafish 執行檔時，會自動改用本機引擎（更快）。
- 影片：可用頁面上的選擇檔案載入本機 video.mp4。

## 引擎與授權
- Pikafish（GPL-3.0）：https://github.com/official-pikafish/Pikafish
- 瀏覽器版 WebAssembly 建置：https://github.com/brianhliou/pikafish-wasm （v0.1.0，主線 2026-09，權重與 2026-09-06 正式版相同）
- 神經網路權重授權見 engine/NNUE-License.md（禁止商業用途）
- coi-serviceworker（MIT）：讓 GitHub Pages 啟用跨源隔離以支援多執行緒
