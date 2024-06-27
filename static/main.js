document.addEventListener('DOMContentLoaded', (event) => {
    const affiliationCells = document.querySelectorAll('.affiliation-cell');
    console.log(affiliationCells);
    affiliationCells.forEach((cell) => {
      let text = cell.textContent;
      // Remove the brackets and quotes
      text = text.replace(/^\['|'\]$/g, "");
      // Update the cell text
      cell.textContent = text;
    });
  
    let yearCells = document.querySelectorAll('.year-cell');
    yearCells.forEach((cell) => {
        let year = parseInt(cell.textContent, 10);
        cell.textContent = isNaN(year) ? '' : year;
    });
  
    let citationCells = document.querySelectorAll('.citation-cell');
    citationCells.forEach((cell) => {
        let citations = parseInt(cell.textContent, 10);
        cell.textContent = isNaN(citations) ? '' : citations;
    });
});
