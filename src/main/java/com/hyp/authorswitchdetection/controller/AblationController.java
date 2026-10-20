package com.hyp.authorswitchdetection.controller;

import com.hyp.authorswitchdetection.model.User;
import com.hyp.authorswitchdetection.service.CurrentUserService;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The ablation study on the web: lists the .csv files that Ablation.java / RetrievalMain write into
 * app.ablation.dir and shows the one that is chosen as a table. Nothing is computed here, the page only
 * reads what the study wrote (so it works while a study is still running: the files grow row by row).
 */
@Controller
public class AblationController {

    @Value("${app.ablation.dir:data/ablation}")
    private String ablationDir;

    private final CurrentUserService currentUserService;

    public AblationController(CurrentUserService currentUserService) {
        this.currentUserService = currentUserService;
    }

    @GetMapping("/ablation")
    public String show(@RequestParam(required = false) String file, Model model, HttpSession session) {

        User user = currentUserService.getCurrentUser(session).orElse(null);
        model.addAttribute("currentUser", user);
        model.addAttribute("dir", ablationDir);

        File[] found = new File(ablationDir).listFiles((d, name) -> name.toLowerCase().endsWith(".csv"));
        List<String> names = new ArrayList<>();
        if (found != null) {
            Arrays.sort(found, (a, b) -> a.getName().compareTo(b.getName()));
            for (File f : found) {
                names.add(f.getName());
            }
        }
        model.addAttribute("files", names);

        String selected = file != null && names.contains(file) ? file : (names.isEmpty() ? null : names.get(0));
        model.addAttribute("selected", selected);

        if (selected != null) {
            try {
                List<List<String>> rows = parseCsv(Files.readString(new File(ablationDir, selected).toPath(),
                        StandardCharsets.UTF_8));
                if (!rows.isEmpty()) {
                    model.addAttribute("header", rows.get(0));
                    model.addAttribute("rows", rows.subList(1, rows.size()));
                }
                File report = new File(ablationDir, "ablation_report.md");
                model.addAttribute("hasReport", report.isFile());
            } catch (IOException e) {
                model.addAttribute("error", "Could not read " + selected + ": " + e.getMessage());
            }
        }
        return "ablation";
    }

    /** A small csv reader: quoted fields (with "" for a quote and commas inside) are what the study writes. */
    static List<List<String>> parseCsv(String text) {

        List<List<String>> rows = new ArrayList<>();
        List<String> row = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean quoted = false;

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < text.length() && text.charAt(i + 1) == '"') {
                        field.append('"');
                        i++;
                    } else {
                        quoted = false;
                    }
                } else {
                    field.append(c);
                }
            } else if (c == '"') {
                quoted = true;
            } else if (c == ',') {
                row.add(field.toString());
                field.setLength(0);
            } else if (c == '\n' || c == '\r') {
                if (c == '\r' && i + 1 < text.length() && text.charAt(i + 1) == '\n') {
                    i++;
                }
                row.add(field.toString());
                field.setLength(0);
                if (!(row.size() == 1 && row.get(0).isEmpty())) {
                    rows.add(row);
                }
                row = new ArrayList<>();
            } else {
                field.append(c);
            }
        }
        if (field.length() > 0 || !row.isEmpty()) {
            row.add(field.toString());
            rows.add(row);
        }
        return rows;
    }
}
