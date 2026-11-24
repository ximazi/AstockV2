(() => {
  const main = document.querySelector('main');
  const expires = Date.parse(main.dataset.validUntil);
  function checkExpiry() {
    if (main.dataset.live !== 'true' || !Number.isFinite(expires) || Date.now() < expires) return;
    main.dataset.live = 'false';
    document.querySelector('.notice').classList.remove('live');
    document.querySelector('.notice strong').textContent = '盘中快照已过期；下方行情及持仓仅为历史观察，需重新读取';
    const priceMode = document.querySelector('#price-review .price-mode');
    if (priceMode) priceMode.textContent = '历史价格观察，已过期，非当前建议';
    document.querySelector('#candidates .section-heading strong').textContent = '模拟参与复核已撤回';
    document.querySelector('#candidates tbody').replaceChildren();
    const selection = document.querySelector('#tail-selection');
    if (selection) {
      selection.querySelector('.selection-mode').textContent = '历史观察，已过期，复核已撤回';
      selection.querySelector('.selection-focus').textContent = '0';
      selection.querySelector('.selection-reviews').textContent = '0';
      selection.querySelector('.selection-notice').textContent = '盘中快照已过期，尾盘策略记录已清空；需重新读取有效结果。';
      selection.querySelectorAll('tbody').forEach(body => body.replaceChildren());
      const more = selection.querySelector('.selection-more');
      if (more) more.remove();
    }
    const observation = document.querySelector('#tail-observation .observation-mode');
    if (observation) observation.textContent = '历史观察，已过期，非当前判断';
  }
  checkExpiry();
  setInterval(checkExpiry, 1000);
  document.addEventListener('visibilitychange', checkExpiry);
})();
