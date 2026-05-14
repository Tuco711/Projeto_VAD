window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.formatDate = function(value) {
    if (value === null || value === undefined) return '';
    var d = new Date(value);
    if (isNaN(d)) return String(value);
    var day = String(d.getDate()).padStart(2, '0');
    var month = String(d.getMonth() + 1).padStart(2, '0');
    var year = d.getFullYear();
    return day + '/' + month + '/' + year;
};
