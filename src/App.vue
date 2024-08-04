<template>
  <div id="app">
    <h1>Urdu OCR System</h1>
    <FileUpload @fileUploaded="setExtractedText" />
    <TextDisplay :text="extractedText" />
  </div>
</template>

<script>
import FileUpload from "./components/FileUpload.vue";
import TextDisplay from "./components/TextDisplay.vue";

export default {
  components: {
    FileUpload,
    TextDisplay,
  },
  data() {
    return {
      extractedText: "",
    };
  },
  methods: {
    setExtractedText(text) {
      this.extractedText = text;
    },
  },
};
</script>

<style>
/* Add some global styling */
#app {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  text-align: center;
}
</style>
