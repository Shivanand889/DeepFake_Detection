import React, { useState, useEffect } from "react";
import "./Homepage.css";

const usePyth = (selectedFile, setFakeScore) => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch('http://127.0.0.1:5000/api/model', {
          method: 'POST',
          body: formData,
        });
        const jsondata = await response.json();
        setData(jsondata);
      } catch (error) {
        console.log(error);
      }
    };

    fetchData();
  }, [selectedFile]);

  useEffect(() => {
    if (data) {
      const fake = parseFloat(data.me);
      if (fake < 0.5) {
        setFakeScore('Fake');
      } else {
        setFakeScore('Real');
      }
    }
  }, [data, setFakeScore]);

  return data;
};

const HomePage = () => {
  const [file, setFile] = useState(null);
  const [fakeScore, setFakeScore] = useState(null);
  const data = usePyth(file, setFakeScore);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
  };

  return (
    <div className="container">
      <header>
        <h1>Detect Fake Image or Video</h1>
      </header>
      <main>
        <label htmlFor="fileInput">Choose your file: </label>
        <input
          type="file"
          id="fileInput"
          accept=".png, .jpg, .jpeg"
          onChange={handleFileChange}
        />
        {file && (
          <div className="file-preview">
            {file.type.startsWith("image/") ? (
              <img src={URL.createObjectURL(file)} alt="" />
            ) : (
              <video controls>
                <source src={URL.createObjectURL(file)} type={file.type} />
                Your browser does not support the video tag.
              </video>
            )}
            {fakeScore !== null && (
              <p>Image is : {fakeScore}</p>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default HomePage;
