<template>
    <div>
      <h2>Upload File</h2>
      <input type="file" @change="onFileChange" />
      <button @click="uploadFile" :disabled="!selectedFile">Upload</button>
    </div>
  </template>
  
  <script>
  import axios from "axios";
  
  export default {
    data() {
      return {
        selectedFile: null,
      };
    },
    methods: {
      onFileChange(event) {
        this.selectedFile = event.target.files[0];
      },
      async uploadFile() {
        if (!this.selectedFile) return;
        const formData = new FormData();
        formData.append("file", this.selectedFile);
  
        try {
          const response = await axios.post(
            "http://localhost:5000/upload",
            formData,
            {
              headers: {
                "Content-Type": "multipart/form-data",
              },
            }
          );
          this.$emit("fileUploaded", response.data.text);
        } catch (error) {
          console.error("Error uploading file:", error);
        }
      },
    },
  };
  </script>
  
  <style scoped>
  /* Add some styling */
  button {
    margin-top: 10px;
  }
  </style>
  