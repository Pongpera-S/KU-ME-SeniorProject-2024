function addNodeInput() {
    const nodeCount = document.querySelectorAll('#node-coordinates div').length + 1;
    const newNodeDiv = document.createElement('div');
    newNodeDiv.innerHTML = `
        <label for="x[]">Node ${nodeCount} X - Coordinate:</label>
        <input type="text" name="x[]" required>
        <label for="y[]">Y - Coordinate:</label>
        <input type="text" name="y[]" required>
        <button type="button" class="remove-button" onclick="removeInput(this)">Remove</button>
    `;
    document.getElementById('node-coordinates').appendChild(newNodeDiv);
}

function addLdaInput() {
    const elementCount = document.querySelectorAll('#lda-inputs div').length + 1;
    const newLdaDiv = document.createElement('div');
    newLdaDiv.innerHTML = `
        <label for="lda_start_${elementCount}">Element ${elementCount} Start Node:</label>
        <input type="text" name="lda_start_${elementCount}" required>
        <label for="lda_end_${elementCount}">End Node:</label>
        <input type="text" name="lda_end_${elementCount}" required>
        <button type="button" class="remove-button" onclick="removeInput(this)">Remove</button>
    `;
    document.getElementById('lda-inputs').appendChild(newLdaDiv);
}

function addForceInput() {
    const forceCount = document.querySelectorAll('#forces-inputs div').length + 1;
    const newForceDiv = document.createElement('div');
    newForceDiv.innerHTML = `
        <label for="nf[]">Forces Applied At Node (nf):</label>
        <input type="text" name="nf[]" required>
        <label for="Fx[]">Force In X-Direction (Fx):</label>
        <input type="text" name="Fx[]" required>
        <label for="Fy[]">Force In Y-Direction (Fy):</label>
        <input type="text" name="Fy[]" required>
        <button type="button" class="remove-button" onclick="removeInput(this)">Remove</button>
    `;
    document.getElementById('forces-inputs').appendChild(newForceDiv);
}

///////////////////////////////////////////////////////////////////////////////////

function addNbInput() {
    const nbCount = document.querySelectorAll('#nb-inputs div').length + 1;
    const newNbDiv = document.createElement('div');
    newNbDiv.innerHTML = `
        <label for="nb[]">Node without force:</label>
        <input type="text" name="nb[]" required>
        <button type="button" class="remove-button" onclick="removeInput(this)">Remove</button>
    `;
    document.getElementById('nb-inputs').appendChild(newNbDiv);
}
// Function to toggle visibility of input fields based on radio button selection
document.querySelectorAll('input[name="E_selection"]').forEach((elem) => {
    elem.addEventListener("change", function(event) {
        const value = event.target.value;
        document.getElementById('E_single_input').style.display = value === "single" ? "block" : "none";
        document.getElementById('E_multiple_input').style.display = value === "multiple" ? "block" : "none";
    });
});

document.querySelectorAll('input[name="A_selection"]').forEach((elem) => {
    elem.addEventListener("change", function(event) {
        const value = event.target.value;
        document.getElementById('A_single_input').style.display = value === "single" ? "block" : "none";
        document.getElementById('A_multiple_input').style.display = value === "multiple" ? "block" : "none";
    });
});

function addMaterialInput(type) {
    const elementCount = document.querySelectorAll(`#material-properties-${type} div`).length + 1;
    const newMaterialDiv = document.createElement('div');
    newMaterialDiv.innerHTML = `
        <label for="${type}_${elementCount}">${type === 'E' ? 'Elastic Modulus' : 'Cross-sectional Area'} for Element ${elementCount}:</label>
        <input type="text" id="${type}_${elementCount}" name="${type}[]"><br><br>
        <button type="button" class="remove-button" onclick="removeInput(this)">Remove</button>
    `;
    document.getElementById(`material-properties-${type}`).appendChild(newMaterialDiv);
}

function removeInput(button) {
    const divToRemove = button.parentElement;
    divToRemove.remove();
}