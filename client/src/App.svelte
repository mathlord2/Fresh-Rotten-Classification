<script>
	import { onMount } from 'svelte';
	let filename;
	let dataURL;
	let prediction;

	onMount(() => {
        window.$('.inputFile').change(function(e){
			var reader = new FileReader();
			let file = window.$('.inputFile').prop('files')[0];
			reader.readAsDataURL(file);

            reader.onload = function() {
                dataURL = reader.result;
				filename = e.target.files[0].name;
				
				fetch("/getimg", {
					method: "POST",
					body: JSON.stringify({
						"filename": dataURL
					})
				}).then(pred => pred.text())
				.then(pred => prediction = pred);
            }
        });
    });

	//function submit() {
		//fetch("/run").then(pred => pred.text())
		//.then(pred => prediction = pred)
	//}

	
</script>

<main>
	<h1>Load fruit image</h1>
	<input type="file" id="fruit" class="inputFile" name="fruit" accept=".jpg, .jpeg">
	<label for="fruit">Upload (.jpg or .jpeg) <i class="fa fa-upload"></i></label>
	<br>
	{#if filename != undefined}
		<p>{filename} has been uploaded.</p>
		<img src={dataURL} alt="An image of a fruit :)">
		<br>
	{/if}
	<!--<button on:click={submit}>Run Model</button>-->
	{#if prediction != undefined}
		{#if prediction == "Fresh Apple" || prediction == "Fresh Orange" || prediction == "Fresh Banana"}
			<p>Prediction: Fresh</p>
		{:else}
			<p>Prediction: Rotten</p>
		{/if}
	{/if}
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}

	.inputFile {
		width: 0.1px;
		height: 0.1px;
		opacity: 0;
		overflow: hidden;
		position: absolute;
		z-index: -1;
	}

	.inputFile + label {
		font-size: 1.25em;
		font-weight: 700;
		background-color: rgb(252, 161, 119);
		display: inline-block;
		padding: 20px;
		margin-bottom: 30px;
		cursor: pointer;
	}

	.inputFile:focus + label,
	.inputFile + label:hover {
		background-color: rgb(255, 153, 0);
	}

	img {
		max-height:500px;
		max-width:500px;
		height:auto;
		width:auto;
	}
</style>