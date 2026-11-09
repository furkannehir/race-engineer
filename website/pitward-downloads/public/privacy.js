import { isOptedOut, setOptOut } from './analytics.js';

const control = document.querySelector('#privacy-optout');
const status = document.querySelector('#privacy-status');
const signal = navigator.globalPrivacyControl === true || navigator.doNotTrack === '1';
control.checked = isOptedOut();
function show() {
  status.textContent = signal ? 'Your browser’s privacy signal already keeps analytics off.' : control.checked ? 'Website analytics is off for this browser.' : 'Website page-load and click counts are allowed for this browser.';
}
control.addEventListener('change', () => {
  if (!setOptOut(control.checked)) status.textContent = 'Your browser did not allow this preference to be saved.';
  else show();
});
show();
