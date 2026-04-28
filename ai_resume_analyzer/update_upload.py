import re

with open('app/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure we import filetype
if "import filetype" not in content:
    content = content.replace('import sys, os, time, requests, pytz, random, re, uuid', 'import sys, os, time, requests, pytz, random, re, uuid, filetype')

new_upload_func = """@app.route("/upload-resume", methods=["POST"])
@limiter.limit("50 per day")
@limiter.limit("10 per hour")
def upload_resume():
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]

    if "resume" not in request.files:
        flash("No file selected!", "error")
        return redirect(url_for("upload"))

    files = request.files.getlist("resume")
    if not files or files[0].filename == "":
        flash("No file selected!", "error")
        return redirect(url_for("upload"))

    success_count = 0
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    last_result = None

    for file in files:
        if not allowed_file(file.filename):
            flash(f"Skipped {file.filename}: Only PDF, DOC, DOCX files are allowed!", "warning")
            continue

        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)

        if file_length > MAX_FILE_SIZE:
            flash(f"Skipped {file.filename}: File too large! Max file size is 28MB.", "warning")
            continue

        # --- MALWARE / PAYLOAD PROTECTION (Magic Bytes Check) ---
        kind = filetype.guess(file.stream)
        file.seek(0) # reset pointer after reading magic bytes
        
        if kind is None or kind.mime not in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            flash(f"Security Alert: Rejected {file.filename}. Invalid file signature detected.", "error")
            continue
            
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        file_hash = generate_hash(filepath)

        file_integrity_collection.insert_one({
            "user_id": user_id,
            "filename": filename,
            "filehash": file_hash,
            "uploaded_at": datetime.now()
        })

        status = "Pending"
        match_score = None
        start_time = time.time()
        process_time = 0
        result = {}
        extracted_text = ""

        try:
            parser_url = os.getenv("PARSERAI_URL") or "https://dyno0126-cv-mind-analyzer.hf.space/upload-analyze"

            with open(filepath, "rb") as f:
                upload_files = {"file": (filename, f)}
                response = requests.post(parser_url, files=upload_files, timeout=60)
                
            if response.status_code == 200:
                result = response.json()
                status = "Success"
                match_score = result.get("final_score", 0)
                extracted_text = result.get("full_resume_text", "")
                
                result["matched_keywords"] = result.get("skills_found", [])
                result["missing_keywords"] = result.get("missing_skills", [])
            else:
                status = "Error"
                print(f"[ERROR] API analysis failed with status {response.status_code}: {response.text}")
                
            process_time = round(time.time() - start_time, 2)
            
        except Exception as e:
            process_time = round(time.time() - start_time, 2)
            status = "Error"
            print(f"[ERROR] Analysis failed: {e}")

        extracted_name = extract_candidate_name(extracted_text)
        final_candidate_name = extracted_name if extracted_name else session.get("name", "User")

        if status == "Success":
            last_result = {
                "name": final_candidate_name,
                "predicted_domain": result.get("predicted_domain", "Unknown"),
                "confidence": round(result.get("confidence", 0) * 100, 1),
                "final_score": match_score,
                "missing_skills": result.get("missing_keywords", []),
                "strengths": result.get("matched_keywords", []),
                "suggestions": result.get("suggestions", []),
                "latency_ms": round(process_time * 1000)
            }
            success_count += 1

        activity_collection.insert_one({
            "user_id": user_id,
            "candidate_name": final_candidate_name,
            "upload_date": datetime.now(),
            "status": status,
            "match_score": match_score,
            "file_name": filename,
            "predicted_domain": result.get("predicted_domain", "Unknown"),
            "confidence": result.get("confidence", 0) * 100,
            "missing_skills": result.get("missing_keywords", []),
            "strengths": result.get("matched_keywords", []),
            "suggestions": result.get("suggestions", []),
            "latency_ms": round(process_time * 1000)
        })

        stats_collection.update_one(
            {"user_id": user_id},
            {
                "$inc": {"total_resumes": 1},
                "$set": {"processing_time": process_time, "updated_at": datetime.now()}
            },
            upsert=True
        )

        if status == "Success":
            stats_collection.update_one(
                {"user_id": user_id},
                {"$inc": {"parsed_success": 1}},
                upsert=True
            )

    if last_result:
        session['last_result'] = last_result

    scores = list(activity_collection.find(
        {"user_id": user_id, "match_score": {"$ne": None}},
        {"match_score": 1}
    ))

    avg = round(sum([s["match_score"] for s in scores]) / len(scores), 2) if scores else 0

    stats_collection.update_one(
        {"user_id": user_id},
        {"$set": {"avg_match_score": avg}}
    )

    if success_count > 0:
        flash(f"Successfully processed {success_count} resume(s).", "success")
    else:
        flash("No resumes were successfully processed.", "error")
        
    return redirect(url_for("dashboard"))
"""

pattern = re.compile(r'@app\.route\("/upload-resume", methods=\["POST"\]\)\ndef upload_resume\(\):.*?return redirect\(url_for\("dashboard"\)\)', re.DOTALL)
new_content = pattern.sub(new_upload_func, content)

with open('app/app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
