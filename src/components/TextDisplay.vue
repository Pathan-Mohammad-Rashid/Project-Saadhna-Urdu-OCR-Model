<!-- <template>
    <div>
      <h2>Extracted Text</h2>
      <textarea v-model="text" rows="10" cols="50" readonly></textarea>
      <button @click="downloadText" :disabled="!text">Download Text</button>
    </div>
  </template>
  
  <script>
  export default {
    props: ["text"],
    methods: {
      downloadText() {
        const blob = new Blob([this.text], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "extracted_text.txt";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      },
    },
  };
  </script>
  
  <style scoped>
  /* Add some styling */
  textarea {
    width: 100%;
    margin-top: 10px;
  }
  button {
    margin-top: 10px;
  }
  </style>
   -->

   <template>
    <div>
      <h2>Extracted Text</h2>
      <textarea :value="text" rows="10" cols="50" readonly></textarea>
      <button @click="downloadText" :disabled="!text">Download Text</button>
    </div>
  </template>
  
  <script>
  export default {
    props: ["text"],
    methods: {
      downloadText() {
        const blob = new Blob([this.text], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "extracted_text.txt";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      },
    },
  };
  </script>
  
  <style scoped>
  /* Add some styling */
  textarea {
    width: 100%;
    margin-top: 10px;
  }
  button {
    margin-top: 10px;
  }
  </style>
  
